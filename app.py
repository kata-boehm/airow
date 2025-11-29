import pandas as pd
import numpy as np
from datetime import timedelta
import io
import os
import base64
from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table
import plotly.graph_objects as go
from dash.dcc import send_data_frame

# === DASH APP SETUP ===
app = Dash(__name__)
app.title = "Data Annotator"

# === LAYOUT ===
app.layout = html.Div([
    dcc.Store(id="uploaded-filename", data="uploaded_file"),
    dcc.Store(id="uploaded-contents", data=None),
    dcc.Store(id="selected-store", data=[]),
    dcc.Store(id="processed-df-store"),

    html.Div(id="click-output"),

    html.Button("Download Selected Timestamps", id="export-button", disabled=True),
    dcc.Download(id="download-file"),

    html.H2("📈 Interactive Time Series Annotator with Power Zones"),

    dcc.Upload(
        id="upload-data",
        children=html.Div(["📂 Drag and Drop or Click to Upload CSV"]),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed",
            "borderRadius": "5px", "textAlign": "center",
            "marginBottom": "10px"
        },
        multiple=False,
        accept=".csv"
    ),

    html.Div([
        html.Label("FTP (Functional Threshold Power):"),
        dcc.Input(id="ftp-input", type="number", value=230, min=50, max=1000, step=5),

        html.Label("Watt Threshold:"),
        dcc.Input(id="watt-threshold-input", type="number", value=11, min=1, max=50, step=1),
    ], style={"marginBottom": "10px", "display": "flex", "gap": "20px"}),

    html.Div([
        html.Button("Detect Start Points", id="auto-detect-button", style={"marginRight": "10px"}),
        html.Button("Labels Modell", id="model-labels-button"),
        dcc.Download(id="download-model-labels")
    ], style={"marginBottom": "10px"}),

    dcc.Graph(id="timeseries-plot", config={"displayModeBar": True, "scrollZoom": True}),

    html.H4("Selected Timestamps:"),
    dash_table.DataTable(
        id="timestamp-table",
        columns=[
            {"name": "Label", "id": "label"},
            {"name": "Timestamp", "id": "timestamp"},
        ],
        style_table={"maxHeight": "300px", "overflowY": "scroll"},
        editable=False,
        row_deletable=True,
    ),
])


# === HELPER: FULL PROCESSING PIPELINE ===
def process_full_pipeline(contents, ftp_value, watt_threshold_value):
    """Process uploaded CSV through entire pipeline"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: 'timestamp'")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

    # Only process if needed
    if "interval_zone_type" not in df.columns:
        if "power" not in df.columns:
            raise ValueError("Missing required column: 'power'")

        df = process_csv_df(df, ftp=ftp_value, watt_drop=watt_threshold_value)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = group_true_values(df)
        df = enforce_consecutive_intervals(df)
        df = assign_dominant_zone_type_per_interval(df)
        df = merge_consecutive_same_zone_intervals(df)
        df = detect_and_invalidate_stop_resume_events(df)
        df = merge_short_intervals(df)
        df = reassign_first_n_seconds(df, seconds=60)
        df = merge_consecutive_same_zone_intervals(df)

    return df


# === HELPER: BUILD FIGURE ===
def build_figure(df, updated_points):
    """Build the plotly figure from dataframe and selected points"""
    fig = go.Figure()

    # Add invisible click target layer
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['power_roll_avg'],
        mode="markers",
        marker=dict(size=8, color='rgba(0,0,0,0.01)'),
        name="Click Target",
        hovertemplate="%{x|%H:%M:%S}<br>%{y:.0f} W",
        visible=True,
        showlegend=False,
        opacity=0.5
    ))

    # Color map for zones
    color_map = {
        "Zone1": "#00008B", "Zone2": "#5dade2", "Zone3": "#229954",
        "Zone4": "#f1c40f", "Zone5": "#e67e22", "Zone6": "#e74c3c", "Zone7": "#7b241c",
        "Unclassified": "gray"
    }

    # Add colored segments for each zone
    for _, group in df.groupby((df["interval_zone_type"] != df["interval_zone_type"].shift()).cumsum()):
        zone = group["interval_zone_type"].iloc[0]
        color = color_map.get(zone, "gray")
        fig.add_trace(go.Scatter(
            x=group["timestamp"],
            y=group["power_roll_avg"],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip"
        ))

    # Add selected points if any exist
    if updated_points:
        selected_df = pd.DataFrame(updated_points)
        selected_df["timestamp"] = pd.to_datetime(selected_df["timestamp"], format="mixed", errors="coerce")
        selected_df = selected_df.dropna(subset=["timestamp"])
        
        # Map timestamps to y-values from dataframe
        selected_df["y_value"] = selected_df["timestamp"].apply(
            lambda ts: df.loc[df["timestamp"] == ts, "power_roll_avg"].values[0]
            if ts in df["timestamp"].values else None
        )
        
        fig.add_trace(go.Scatter(
            x=selected_df["timestamp"],
            y=selected_df["y_value"],
            mode="markers+text",
            text=selected_df["label"],
            textposition="top center",
            marker=dict(color="black", size=10),
            name="Selected Points"
        ))

    # Add legend entries for each zone
    for zone, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=color, width=3),
            name=zone
        ))

    fig.update_layout(
        title="Power Zone Visualization",
        xaxis_title="Time",
        yaxis_title="Power",
        hovermode='closest'
    )

    return fig


# === MAIN CALLBACK ===
@app.callback(
    Output("timeseries-plot", "figure"),
    Output("export-button", "disabled"),
    Output("timestamp-table", "data"),
    Output("selected-store", "data"),
    Output("download-file", "data"),
    Output("click-output", "children"),
    Output("processed-df-store", "data"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename"),
    Input("ftp-input", "value"),
    Input("watt-threshold-input", "value"),
    Input("timeseries-plot", "clickData"),
    Input("export-button", "n_clicks"),
    Input("timestamp-table", "data"),
    Input("auto-detect-button", "n_clicks"),
    State("selected-store", "data"),
    State("processed-df-store", "data"),
    prevent_initial_call=True
)
def update_graph_and_handle_click(contents, uploaded_filename, ftp_value, watt_threshold_value,
                                   clickData, export_clicks, current_table, auto_detect_clicks,
                                   selected_store, stored_df_json):

    triggered = ctx.triggered_id
    
    # Determine if we need to reprocess the entire CSV
    needs_reprocessing = triggered in ["upload-data", "ftp-input", "watt-threshold-input"]
    
    # === PATH 1: REPROCESS EVERYTHING ===
    if needs_reprocessing:
        if contents is None:
            return {}, True, [], [], None, "", None
        
        try:
            # Process the entire pipeline
            df = process_full_pipeline(contents, ftp_value, watt_threshold_value)
            
            # Store as JSON using 'split' orientation for better performance
            df_json = df.to_json(date_format='iso', orient='split')
            
            # Check for pre-existing Manual_Timestamps column
            updated_points = []
            if "Manual_Timestamps" in df.columns:
                manual_timestamps = df.loc[df["Manual_Timestamps"] == True, "timestamp"].dropna().dt.floor("s")
                updated_points = [{"timestamp": str(ts), "y_value": None, "label": i+1} 
                                  for i, ts in enumerate(manual_timestamps)]
            
        except Exception as e:
            return {}, True, [], [], None, f"Error: {str(e)}", None
    
    # === PATH 2: REUSE STORED DATA (FAST PATH) ===
    else:
        if stored_df_json is None:
            return {}, True, [], [], None, "No data loaded", None
        
        # Read from stored JSON (much faster than reprocessing)
        df = pd.read_json(stored_df_json, orient='split')
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_json = stored_df_json  # Reuse without modification
        
        # Start with current table data or empty list
        updated_points = current_table.copy() if current_table else []
        
        # Handle auto-detect button
        if triggered == "auto-detect-button":
            start_times = df.loc[df["final_group_start"] == True, "timestamp"].dropna().dt.floor("s")
            updated_points = [{"timestamp": str(ts), "y_value": None} for ts in start_times]
        
        # Handle graph clicks (add/remove points)
        elif triggered == "timeseries-plot" and clickData:
            timestamp_clicked = clickData["points"][0]["x"]
            y_clicked = clickData["points"][0]["y"]
            
            # Toggle: remove if exists, add if new
            if any(p["timestamp"] == timestamp_clicked for p in updated_points):
                updated_points = [p for p in updated_points if p["timestamp"] != timestamp_clicked]
            else:
                updated_points.append({"timestamp": timestamp_clicked, "y_value": y_clicked})
        
        # Handle row deletion from table
        elif triggered == "timestamp-table":
            # current_table already reflects deletions from the UI
            updated_points = current_table
    
    # === COMMON PROCESSING FOR BOTH PATHS ===
    
    # Sort points by timestamp and assign labels
    updated_points = sorted(updated_points, key=lambda x: x["timestamp"])
    for i, point in enumerate(updated_points):
        point["label"] = i + 1
    
    # Build the figure
    fig = build_figure(df, updated_points)
    
    # Prepare table data for display
    table_data = []
    for p in updated_points:
        timestamp_display = pd.to_datetime(p["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        table_data.append({"label": p["label"], "timestamp": timestamp_display})
    
    # Disable export button if no points selected
    export_disabled = len(updated_points) == 0
    
    # Handle export
    download_data = None
    if triggered == "export-button" and updated_points:
        timestamps_to_mark = pd.to_datetime([p["timestamp"] for p in updated_points])
        df["Manual_Timestamps"] = df["timestamp"].dt.floor("s").isin(timestamps_to_mark.floor("s"))
        
        # Determine export filename
        if uploaded_filename and uploaded_filename.lower().endswith(".csv"):
            base_name = uploaded_filename[:-4]
        else:
            base_name = uploaded_filename or "export"
        
        export_name = f"{base_name}_with_manual_labels.csv"
        download_data = send_data_frame(df.to_csv, export_name, index=False)
    
    return fig, export_disabled, table_data, updated_points, download_data, "", df_json


# === CALLBACK: EXPORT MODEL LABELS ===
@app.callback(
    Output("download-model-labels", "data"),
    Input("model-labels-button", "n_clicks"),
    State("processed-df-store", "data"),
    prevent_initial_call=True
)
def export_model_labels(n_clicks, stored_df_json):
    if not stored_df_json:
        return None
    
    df = pd.read_json(stored_df_json, orient='split')
    df = df[df["final_group_start"] == True]
    
    return send_data_frame(df[["timestamp", "interval_group_id"]].to_csv, "model_labels.csv", index=False)


# === SUPPORTING FUNCTIONS ===
def process_csv_df(df, ftp, seven_zone_model=True, window_size=5, use_roll_avg=True,
                   watt_drop=11, extend_gradient=True):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset='timestamp', keep='first')

    expected = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='1s')
    df = df.set_index('timestamp').reindex(expected).reset_index().rename(columns={'index': 'timestamp'})
    df['power_roll_avg'] = df['power'].rolling(window_size, center=True, min_periods=1).mean()
    df['delta_time'] = df['timestamp'].diff().dt.total_seconds()
    df['power_derivative'] = df['power_roll_avg'].diff() / df['delta_time']

    power_slope_threshold = watt_drop
    df['interval_start_candidate'] = (df['power_derivative'] > power_slope_threshold).fillna(False)
    df['interval_end_candidate'] = (df['power_derivative'] < -power_slope_threshold).fillna(False)

    # Extend confirmed_interval_start backward where the gradient is still positive
    if extend_gradient:
        # Extend interval_start_candidate backward if avg gradient in prior 5s is still positive
        df['extended_start'] = False
        in_interval = False
        for i in df.index[::-1]:  # reverse loop
            if df.at[i, 'interval_start_candidate']:
                in_interval = True
            elif in_interval:
                current_time = df.at[i, 'timestamp']
                past_time = current_time - timedelta(seconds=5)

                past_segment = df[(df['timestamp'] >= past_time) & (df['timestamp'] < current_time)]
                if not past_segment.empty and past_segment['power_derivative'].mean() > 0:
                    df.at[i, 'extended_start'] = True
                else:
                    in_interval = False

        # Extend interval_end_candidate forward where avg gradient in next 5s is still negative
        df['extended_end'] = False
        in_interval = False
        for i in df.index:
            if df.at[i, 'interval_end_candidate']:
                in_interval = True
            elif in_interval:
                current_time = df.at[i, 'timestamp']
                future_time = current_time + timedelta(seconds=10)

                future_segment = df[(df['timestamp'] > current_time) & (df['timestamp'] <= future_time)]
                if not future_segment.empty and future_segment['power_derivative'].mean() < 0:
                    df.at[i, 'extended_end'] = True
                else:
                    in_interval = False

        # Merge with original
        df['interval_start_candidate'] = df['interval_start_candidate'] | df['extended_start']
        df['interval_end_candidate'] = df['interval_end_candidate'] | df['extended_end']
        df.drop(columns=['extended_start', 'extended_end'], inplace=True)

    # Choose power data for classification
    df['power_for_classification'] = df['power_roll_avg'] if use_roll_avg else df['power']

    # Power Zones
    if seven_zone_model:
        power_bins = [0, 55, 75, 90, 105, 120, 150, float('inf')]
        power_labels = ['Zone1', 'Zone2', 'Zone3', 'Zone4', 'Zone5', 'Zone6', 'Zone7']
    else:
        power_bins = [0, 55, 75, 90, 105, 120, float('inf')]
        power_labels = ['Zone1', 'Zone2', 'Zone3', 'Zone4', 'Zone5', 'Zone6']

    df['Power_Zone'] = pd.cut(df['power_for_classification'] / ftp * 100, bins=power_bins, labels=power_labels)
    df['Interval_Type'] = df['Power_Zone'].astype(str)

    return df


def group_true_values(df, start_column='interval_start_candidate', end_column='interval_end_candidate'):
    # Shifted versions for neighborhood check
    prev1_end = df[start_column].shift(1, fill_value=False)
    next1_end = df[end_column].shift(-1, fill_value=False)
    prev2_end = df[start_column].shift(2, fill_value=False)
    next2_end = df[end_column].shift(-2, fill_value=False)

    # Identify groups of True values in the sequence
    group_start = (df[start_column] == True) & (prev1_end == False) & (prev2_end == False)
    group_end = (df[end_column] == True) & (next1_end == False) & (next2_end == False)

    # Only keep the first occurrence of a group as True
    df['group_first'] = group_start
    df['group_last'] = group_end

    return df


def enforce_consecutive_intervals(df, start_col='group_first', end_col='group_last'):
    df = df.copy()

    # Start: if previous row is a group_end → current row should be a group_start
    prev_end = df[end_col].shift(1, fill_value=False)
    df.loc[prev_end & (~df[start_col]), start_col] = True

    # End: if next row is a group_start → current row should be a group_end
    next_start = df[start_col].shift(-1, fill_value=False)
    df.loc[next_start & (~df[end_col]), end_col] = True

    # Force the very first row to be a start
    df.loc[df.index[0], start_col] = True

    # Force the very last row to be an end
    df.loc[df.index[-1], end_col] = True

    return df


def assign_dominant_zone_type_per_interval(df, start_col='group_first', label_col='Interval_Type'):
    df = df.copy()
    df[label_col] = df[label_col].replace({np.nan: 'Unclassified', 'nan': 'Unclassified'})

    # Assign a group ID to each interval based on cumulative sum of start flags
    df['interval_group_id'] = df[start_col].cumsum()

    # Find the most frequent zone in each group
    dominant_zones = (
        df.groupby('interval_group_id')[label_col]
        .agg(lambda x: x.value_counts().idxmax())
        .rename('interval_zone_type')
    )

    # Merge the result back to the original DataFrame
    df = df.merge(dominant_zones, left_on='interval_group_id', right_index=True)

    return df


def merge_consecutive_same_zone_intervals(df):
    df = df.copy()
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Detect zone changes from row to row
    zone_change = (df['interval_zone_type'] != df['interval_zone_type'].shift()).astype(int)

    # New group ID is just cumulative sum of changes
    df['interval_group_id'] = zone_change.cumsum()

    # Re-assign interval_zone_type per new group (ensures consistent label)
    new_zones = (
        df.groupby('interval_group_id')['interval_zone_type']
        .agg(lambda x: x.value_counts().idxmax())
    )
    df['interval_zone_type'] = df['interval_group_id'].map(new_zones)

    # Initialize new columns
    df['final_group_start'] = False
    df['final_group_end'] = False

    # Mark first and last index of each group
    group_indices = df.groupby('interval_group_id').agg(first_idx=('timestamp', 'idxmin'),
                                                        last_idx=('timestamp', 'idxmax'))

    df.loc[group_indices['first_idx'], 'final_group_start'] = True
    df.loc[group_indices['last_idx'], 'final_group_end'] = True

    return df


def detect_and_invalidate_stop_resume_events(df, window_seconds=20, drop_threshold=100, recovery_margin=10):
    """
    Detects stop-resume events based on sudden power drops and quick recoveries,
    and directly updates the DataFrame:
    - Sets final_group_start and final_group_end to False within drop windows
    - Reassigns interval_group_id and interval_zone_type to the last valid value before the drop
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["power_roll_avg"])

    power = df["power_roll_avg"].values
    timestamps = df["timestamp"].values

    for i in range(len(df) - window_seconds):
        window_power = power[i:i + window_seconds]
        window_time = timestamps[i:i + window_seconds]

        pre_value = window_power[0]
        min_idx = window_power.argmin()
        min_value = window_power[min_idx]
        post_value = window_power[-1]

        drop = pre_value - min_value
        recovered = abs(post_value - pre_value) <= recovery_margin

        if drop >= drop_threshold and recovered and 1 < min_idx < window_seconds - 2:
            start = window_time[0]
            end = window_time[-1]

            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)

            # Get last known interval metadata before the drop
            prior = df[df["timestamp"] < start]
            fill_zone = prior["interval_zone_type"].dropna().iloc[-1] if not prior["interval_zone_type"].dropna().empty else None
            fill_group = prior["interval_group_id"].dropna().iloc[-1] if not prior["interval_group_id"].dropna().empty else None

            df.loc[mask, "final_group_start"] = False
            df.loc[mask, "final_group_end"] = False
            df.loc[mask, "interval_group_id"] = fill_group
            df.loc[mask, "interval_zone_type"] = fill_zone

    return df


def merge_short_intervals(df, min_duration_seconds=5):
    """
    Reassigns intervals shorter than min_duration_seconds to neighboring intervals
    based on which neighbor has closer average power_roll_avg.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Ensure group ID is consistent and not NaN
    df["interval_group_id"] = df["interval_group_id"].ffill()

    grouped = df.groupby("interval_group_id")

    for group_id, group_df in grouped:
        duration = (group_df["timestamp"].iloc[-1] - group_df["timestamp"].iloc[0]).total_seconds()
        if duration >= min_duration_seconds:
            continue  # skip long enough intervals

        # Mean power of this short interval
        this_mask = (df["timestamp"] >= group_df["timestamp"].iloc[0]) & (df["timestamp"] <= group_df["timestamp"].iloc[-1])
        this_mean = group_df["power_roll_avg"].mean()

        # 5 seconds before
        before_mask = (df["timestamp"] < group_df["timestamp"].iloc[0]) & \
                      (df["timestamp"] >= group_df["timestamp"].iloc[0] - pd.Timedelta(seconds=5))
        before_mean = df.loc[before_mask, "power_roll_avg"].mean()
        before_group = df.loc[before_mask, "interval_group_id"].dropna().iloc[-1] if not df.loc[before_mask].empty else None
        before_zone = df.loc[before_mask, "interval_zone_type"].dropna().iloc[-1] if not df.loc[before_mask].empty else None

        # 5 seconds after
        after_mask = (df["timestamp"] > group_df["timestamp"].iloc[-1]) & \
                     (df["timestamp"] <= group_df["timestamp"].iloc[-1] + pd.Timedelta(seconds=5))
        after_mean = df.loc[after_mask, "power_roll_avg"].mean()
        after_group = df.loc[after_mask, "interval_group_id"].dropna().iloc[0] if not df.loc[after_mask].empty else None
        after_zone = df.loc[after_mask, "interval_zone_type"].dropna().iloc[0] if not df.loc[after_mask].empty else None

        # Choose the closest in mean power
        if pd.isna(before_mean) and pd.isna(after_mean):
            continue  # nowhere to merge to

        if pd.isna(before_mean):
            assign_group, assign_zone = after_group, after_zone
        elif pd.isna(after_mean):
            assign_group, assign_zone = before_group, before_zone
        else:
            if abs(this_mean - before_mean) <= abs(this_mean - after_mean):
                assign_group, assign_zone = before_group, before_zone
            else:
                assign_group, assign_zone = after_group, after_zone

        # Apply reassignment
        df.loc[this_mask, "interval_group_id"] = assign_group
        df.loc[this_mask, "interval_zone_type"] = assign_zone
        df.loc[this_mask, "final_group_start"] = False
        df.loc[this_mask, "final_group_end"] = False

    return df


def reassign_first_n_seconds(df, seconds=30):
    if df.empty:
        return df

    start_time = df["timestamp"].iloc[0]
    cutoff_time = start_time + pd.Timedelta(seconds=seconds)

    # Find first valid zone after cutoff
    future_mask = df["timestamp"] > cutoff_time
    valid_future = df.loc[future_mask & df["interval_zone_type"].notna()]
    valid_future = valid_future[valid_future["interval_zone_type"] != "Unclassified"]

    if valid_future.empty:

        return df  # nothing to assign from

    new_zone = valid_future.iloc[0]["interval_zone_type"]

    # Apply reassignment
    df.loc[df["timestamp"] <= cutoff_time, "interval_zone_type"] = new_zone
    df.loc[valid_future.index[0], "final_group_start"] = False
    df['interval_group_id'] = df['final_group_start'].cumsum()

    return df


# === RUN ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
