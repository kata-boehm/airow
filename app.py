import pandas as pd
import numpy as np
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
    dcc.Store(id="processed-df-store"),  # <-- NEU: Store für DataFrame

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
        dcc.Input(id="watt-threshold-input", type="number", value=20, min=1, max=100, step=1),
    ], style={"marginBottom": "10px", "display": "flex", "gap": "20px"}),

    # ✅ Neue Button-Zeile für "Labels Modell"
    html.Div([
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

# === CALLBACK für Hauptfunktion ===
@app.callback(
    Output("timeseries-plot", "figure"),
    Output("export-button", "disabled"),
    Output("timestamp-table", "data"),
    Output("selected-store", "data"),
    Output("download-file", "data"),
    Output("click-output", "children"),
    Output("processed-df-store", "data"),  # ✅ Neuer Output
    Input("upload-data", "contents"),
    Input("upload-data", "filename"),
    Input("ftp-input", "value"),
    Input("watt-threshold-input", "value"),
    Input("timeseries-plot", "clickData"),
    Input("export-button", "n_clicks"),
    Input("timestamp-table", "data"),
    State("selected-store", "data"),
    prevent_initial_call=True
)
def update_graph_and_handle_click(contents, uploaded_filename, ftp_value, watt_threshold_value, clickData, n_clicks,
                                  current_table, selected_store):
    if contents is None:
        return {}, True, [], [], None, "", None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    if "timestamp" not in df.columns:
        return {}, True, [], [], None, "Missing required column: 'timestamp'", None

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

    # Only process if needed
    if "interval_zone_type" not in df.columns:
        if "power" not in df.columns:
            return {}, True, [], [], None, "Missing required column: 'power'", None

        df = process_csv_df(df, ftp=ftp_value, watt_drop=watt_threshold_value)
        df = group_true_values(df)
        df = enforce_consecutive_intervals(df)
        df = assign_dominant_zone_type_per_interval(df)
        df = merge_consecutive_same_zone_intervals(df)

    # Check for pre-existing Manual_Timestamps column
    manual_timestamps = []
    if "Manual_Timestamps" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
        manual_timestamps = df.loc[df["Manual_Timestamps"] == True, "timestamp"].dropna().dt.floor("s").tolist()

    triggered = ctx.triggered_id
    if triggered == "upload-data" and manual_timestamps:
        updated_points = [{"timestamp": ts, "y_value": None} for ts in manual_timestamps]
    else:
        updated_points = current_table.copy() if current_table else []

    if triggered == "timeseries-plot" and clickData:
        timestamp_clicked = clickData["points"][0]["x"]
        y_clicked = clickData["points"][0]["y"]
        if any(p["timestamp"] == timestamp_clicked for p in updated_points):
            updated_points = [p for p in updated_points if p["timestamp"] != timestamp_clicked]
        else:
            updated_points.append({"timestamp": timestamp_clicked, "y_value": y_clicked})

    updated_points = sorted(updated_points, key=lambda x: x["timestamp"])
    for i, point in enumerate(updated_points):
        point["label"] = i + 1

    fig = go.Figure()

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

    color_map = {
        "Zone1": "#00008B", "Zone2": "#5dade2", "Zone3": "#229954",
        "Zone4": "#f1c40f", "Zone5": "#e67e22", "Zone6": "#e74c3c", "Zone7": "#7b241c",
        "Unclassified": "gray"
    }

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

    if updated_points:
        selected_df = pd.DataFrame(updated_points)
        selected_df["timestamp"] = pd.to_datetime(selected_df["timestamp"], format="mixed", errors="coerce")
        selected_df = selected_df.dropna(subset=["timestamp"])
        selected_df["y_value"] = selected_df["timestamp"].map(
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

    for zone, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=color, width=3),
            name=zone
        ))

    fig.update_layout(title="Power Zone Visualization", xaxis_title="Time", yaxis_title="Power")

    export_disabled = False if updated_points else True
    # Format timestamps for display
    for p in updated_points:
        p["timestamp_display"] = pd.to_datetime(p["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")

    table_data = [{"label": p["label"], "timestamp": p["timestamp_display"]} for p in updated_points]

    download_data = None

    if triggered == "export-button" and updated_points:
        timestamps_to_mark = pd.to_datetime([p["timestamp"] for p in updated_points])
        df["Manual_Timestamps"] = df["timestamp"].dt.floor("s").isin(timestamps_to_mark.floor("s"))

        if uploaded_filename and uploaded_filename.lower().endswith(".csv"):
            base_name = uploaded_filename[:-4]
        else:
            base_name = uploaded_filename or "export"

        export_name = f"{base_name}_with_manual_labels.csv"
        download_data = send_data_frame(df.to_csv, export_name, index=False)

    return fig, export_disabled, table_data, updated_points, download_data, "", df.to_json(date_format='iso')

# === NEUER CALLBACK für "Labels Modell" ===
@app.callback(
    Output("download-model-labels", "data"),
    Input("model-labels-button", "n_clicks"),
    State("processed-df-store", "data"),
    prevent_initial_call=True
)
def export_model_labels(n_clicks, stored_df_json):
    if not stored_df_json:
        return None
    df = pd.read_json(stored_df_json)
    df = df[df["final_group_start"] == True]
    return send_data_frame(df[["timestamp", "interval_group_id"]].to_csv, "model_labels.csv", index=False)

# === SUPPORTING FUNCTIONS ===
def process_csv_df(df, ftp, seven_zone_model=True, window_size=5, use_roll_avg=True,
                   watt_drop=20, extend_gradient=False):
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

    if extend_gradient:
        df['extended_start'] = False
        in_interval = False
        for i in df.index[::-1]:
            if df.at[i, 'interval_start_candidate']:
                in_interval = True
            elif in_interval and df.at[i, 'power_derivative'] > 0:
                df.at[i, 'extended_start'] = True
            else:
                in_interval = False

        df['extended_end'] = False
        in_interval = False
        for i in df.index:
            if df.at[i, 'interval_end_candidate']:
                in_interval = True
            elif in_interval and df.at[i, 'power_derivative'] < 0:
                df.at[i, 'extended_end'] = True
            else:
                in_interval = False

        df['interval_start_candidate'] |= df['extended_start']
        df['interval_end_candidate'] |= df['extended_end']
        df.drop(columns=['extended_start', 'extended_end'], inplace=True)

    df['power_for_classification'] = df['power_roll_avg'] if use_roll_avg else df['power']

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
    """Find the first and last timestamp of an interval"""

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
    """Making sure intervals are consecutive: If an interval ends, another one must start the second after"""

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
    """Classify per detected interval zone"""

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
    """If consecutive intervals have same zone classification, they get merged as one."""

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

# === RUN ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
