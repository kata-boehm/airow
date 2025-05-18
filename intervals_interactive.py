import pandas as pd
import numpy as np
import io
import base64
from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table
import plotly.graph_objects as go

# === DASH APP SETUP ===
app = Dash(__name__)
app.title = "Bike Data Annotator"

# === LAYOUT ===
app.layout = html.Div([
    dcc.Store(id="uploaded-filename", data="uploaded_file"),
    dcc.Store(id="uploaded-contents", data=None),
    html.Div(id="click-output"),
    html.Button("Download Selected Timestamps", id="export-button", disabled=True),
    dcc.Download(id="download-file"),
    html.H2("\ud83d\udcc8 Interactive Time Series Annotator with Power Zones"),

    dcc.Upload(
        id="upload-data",
        children=html.Div(["\ud83d\udcc2 Drag and Drop or Click to Upload CSV"]),
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
        dcc.Input(id="ftp-input", type="number", value=250, min=50, max=1000, step=10),

        html.Label("Watt Threshold:"),
        dcc.Input(id="watt-threshold-input", type="number", value=20, min=1, max=100, step=1),
    ], style={"marginBottom": "10px", "display": "flex", "gap": "20px"}),

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

    dcc.Store(id="selected-store", data=[]),
])

# === CALLBACK ===
from dash import ctx
from dash.dcc import send_data_frame

@app.callback(
    Output("timeseries-plot", "figure"),
    Output("export-button", "disabled"),
    Output("timestamp-table", "data"),
    Output("selected-store", "data"),
    Output("download-file", "data"),
    Output("click-output", "children"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename"),  # ✅ real filename now
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
        return {}, True, [], [], None, ""

    # === Load and process uploaded CSV ===
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    if "timestamp" not in df.columns or "power" not in df.columns:
        return {}, True, [], [], None, "Missing required columns."

    df = process_csv_df(df, ftp=ftp_value, watt_drop=watt_threshold_value)
    df = detect_and_enforce_intervals(df)
    df = assign_and_merge_zone_intervals(df)

    # === Determine what triggered the callback ===
    triggered = ctx.triggered_id
    updated_points = current_table.copy() if current_table else []

    if triggered == "timeseries-plot" and clickData:
        timestamp_clicked = clickData["points"][0]["x"]
        y_clicked = clickData["points"][0]["y"]

        if any(p["timestamp"] == timestamp_clicked for p in updated_points):
            updated_points = [p for p in updated_points if p["timestamp"] != timestamp_clicked]
        else:
            updated_points.append({"timestamp": timestamp_clicked, "y_value": y_clicked})

    # Always relabel
    updated_points = sorted(updated_points, key=lambda x: x["timestamp"])
    for i, point in enumerate(updated_points):
        point["label"] = i + 1

    # === Build Plot ===
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

    # === Export logic ===
    export_disabled = False if updated_points else True
    table_data = [{"label": p["label"], "timestamp": p["timestamp"]} for p in updated_points]
    download_data = None

    if triggered == "export-button" and updated_points:
        df_export = pd.DataFrame(updated_points)[["label", "timestamp"]]
        if uploaded_filename and uploaded_filename.lower().endswith(".csv"):
            base_name = uploaded_filename[:-4]
        else:
            base_name = uploaded_filename or "export"
        export_name = f"{base_name}_labelled.csv"
        download_data = send_data_frame(df_export.to_csv, export_name, index=False)

    message = ""

    return fig, export_disabled, table_data, updated_points, download_data, message



# === SUPPORTING FUNCTIONS ===
def process_csv_df(df, ftp, seven_zone_model=False, window_size=3, use_roll_avg=True,
                   watt_drop=20, extend_gradient=True):
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

def detect_and_enforce_intervals(df, start_column='interval_start_candidate', end_column='interval_end_candidate'):
    df = df.copy()
    group_start = (df[start_column] & ~df[start_column].shift(1, fill_value=False))
    group_end = (df[end_column] & ~df[end_column].shift(-1, fill_value=False))
    df['group_first'] = group_start
    df['group_last'] = group_end
    df.loc[df.index[0], 'group_first'] = True
    df.loc[df.index[-1], 'group_last'] = True
    return df

def assign_dominant_zone_type(df, start_col='group_first', label_col='Interval_Type'):
    df = df.copy()
    df[label_col] = df[label_col].fillna('Unclassified')
    df['interval_group_id'] = df[start_col].cumsum()
    dominant_zones = df.groupby('interval_group_id')[label_col].agg(lambda x: x.mode().iloc[0])
    df['interval_zone_type'] = df['interval_group_id'].map(dominant_zones)
    return df

def assign_and_merge_zone_intervals(df, start_col='group_first', label_col='Interval_Type'):
    df = assign_dominant_zone_type(df, start_col=start_col, label_col=label_col)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    zone_change = (df['interval_zone_type'] != df['interval_zone_type'].shift()).astype(int)
    df['interval_group_id'] = zone_change.cumsum()
    new_zones = df.groupby('interval_group_id')['interval_zone_type'].agg(lambda x: x.mode().iloc[0])
    df['interval_zone_type'] = df['interval_group_id'].map(new_zones)
    df['final_group_start'] = df['interval_group_id'] != df['interval_group_id'].shift().fillna(-1)
    df['final_group_end'] = df['interval_group_id'] != df['interval_group_id'].shift(-1).fillna(-1)
    return df

# === RUN ===
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=False)



