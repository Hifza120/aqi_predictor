import sys 
import joblib 
import numpy as np 
import pandas as pd 
import plotly .graph_objects as go 
import plotly .express as px 
from plotly .subplots import make_subplots 
from pathlib import Path 
from datetime import datetime ,timezone ,timedelta 
from feast import FeatureStore 


ROOT =Path (__file__ ).resolve ().parent .parent 
PIPELINES =ROOT /"pipelines"
FEATURE_REPO =PIPELINES /"feature_repo"
MODELS_DIR =PIPELINES /"models"
EDA_DIR =PIPELINES /"eda_plots"
PARQUET =FEATURE_REPO /"data"/"aqi_features.parquet"


import streamlit as st 




st .set_page_config (
page_title ="Lahore AQI Predictor",
page_icon ="",
layout ="wide",
initial_sidebar_state ="expanded",
)




st .markdown ("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-deep:    #0a0c10;
    --bg-card:    #111520;
    --bg-surface: #181d2a;
    --border:     #252d40;
    --text-prime: #e8eaf2;
    --text-muted: #6b7a99;
    --amber:      #f59e0b;
    --amber-glow: rgba(245,158,11,0.15);
    --green:      #10b981;
    --yellow:     #fbbf24;
    --orange:     #f97316;
    --red:        #ef4444;
    --purple:     #8b5cf6;
    --maroon:     #991b1b;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-deep) !important;
    color: var(--text-prime) !important;
}

.stApp {
    background: var(--bg-deep) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(245,158,11,0.06), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,92,246,0.04), transparent);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-prime) !important; }

/* ── Cards ── */
.aqi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.aqi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
}
.aqi-card:hover { border-color: var(--amber); }

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-sub {
    font-size: 13px;
    color: var(--text-muted);
}

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--amber);
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Alert banner ── */
.alert-banner {
    border-radius: 10px;
    padding: 16px 20px;
    font-weight: 600;
    font-size: 15px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    border: 1px solid;
}

/* ── Forecast horizon cards ── */
.forecast-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.horizon-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--text-muted);
}
.horizon-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
}
.horizon-time {
    font-size: 12px;
    color: var(--text-muted);
}

/* ── Streamlit tweaks ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}
div[data-testid="stMetricValue"] > div {
    font-family: 'Space Mono', monospace;
    font-size: 28px !important;
}
.stSelectbox > div > div {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-prime) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom-color: var(--amber) !important;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--text-prime) !important;
}
</style>
""",unsafe_allow_html =True )




def aqi_color (val ):
    if val is None or (isinstance (val ,float )and np .isnan (val )):return "#6b7a99"
    if val <=50 :return "#10b981"
    if val <=100 :return "#fbbf24"
    if val <=150 :return "#f97316"
    if val <=200 :return "#ef4444"
    if val <=300 :return "#8b5cf6"
    return "#991b1b"

def aqi_label (val ):
    if val is None or (isinstance (val ,float )and np .isnan (val )):return "Unknown"
    if val <=50 :return "Good"
    if val <=100 :return "Moderate"
    if val <=150 :return "Unhealthy (Sensitive)"
    if val <=200 :return "Unhealthy"
    if val <=300 :return "Very Unhealthy"
    return "Hazardous"

def aqi_emoji (val ):
    if val is None :return ""
    if val <=50 :return ""
    if val <=100 :return ""
    if val <=150 :return ""
    if val <=200 :return ""
    if val <=300 :return ""
    return ""

_AXIS =dict (gridcolor ="#252d40",linecolor ="#252d40")

def plotly_dark (**overrides ):
    """Returns a layout dict. Pass any extra keys to merge; xaxis/yaxis are merged not replaced."""
    base =dict (
    plot_bgcolor ="#111520",
    paper_bgcolor ="#111520",
    font =dict (family ="DM Sans",color ="#e8eaf2"),
    xaxis =dict (**_AXIS ),
    yaxis =dict (**_AXIS ),
    margin =dict (l =20 ,r =20 ,t =40 ,b =20 ),
    )
    for k ,v in overrides .items ():
        if k in ("xaxis","yaxis")and isinstance (v ,dict ):
            base [k ]={**_AXIS ,**v }
        elif k =="margin"and isinstance (v ,dict ):
            base [k ]={**base ["margin"],**v }
        else :
            base [k ]=v 
    return base 




@st .cache_data (ttl =3600 )
def load_history ():
    df =pd .read_parquet (PARQUET )
    df ["event_timestamp"]=pd .to_datetime (df ["event_timestamp"],utc =True )
    df =df .sort_values ("event_timestamp").reset_index (drop =True )
    return df 

@st .cache_resource 
def load_models ():
    models ,features ={},{}
    for h in ["24h","48h","72h"]:
        mp =MODELS_DIR /f"best_model_{h }.pkl"
        fp =MODELS_DIR /f"features_{h }.pkl"
        if mp .exists ()and fp .exists ():
            models [h ]=joblib .load (mp )
            features [h ]=joblib .load (fp )
    return models ,features 

def get_live_features ():
    """Try Feast online store, fall back to parquet."""
    try :
        store =FeatureStore (repo_path =str (FEATURE_REPO ))
        online =store .get_online_features (
        features =[
        "aqi_raw:aqi","aqi_raw:pm25","aqi_raw:pm10",
        "aqi_raw:no2","aqi_raw:so2","aqi_raw:co","aqi_raw:o3",
        "aqi_raw:temperature","aqi_raw:humidity",
        "aqi_raw:precipitation","aqi_raw:cloud_cover",
        "aqi_raw:wind_speed","aqi_raw:boundary",
        "aqi_lag:pm25_lag1","aqi_lag:pm25_lag24",
        "aqi_lag:pm25_roll_mean_6","aqi_lag:pm25_roll_mean_24",
        "aqi_lag:aqi_lag1","aqi_lag:aqi_lag24",
        "aqi_lag:aqi_change_rate",
        "aqi_lag:aqi_roll_mean_6","aqi_lag:aqi_roll_mean_24",
        "aqi_time:hour","aqi_time:day","aqi_time:month",
        "aqi_time:year","aqi_time:day_of_week",
        "aqi_time:sin_hour","aqi_time:cos_hour",
        "aqi_time:sin_month","aqi_time:cos_month",
        "aqi_time:sin_day","aqi_time:cos_day",
        "aqi_time:sin_dow","aqi_time:cos_dow",
        ],
        entity_rows =[{"station_id":"lahore_main"}],
        ).to_dict ()
        live ={k .split (":")[-1 ]:v [0 ]for k ,v in online .items ()}
        return live ,"online"
    except Exception :
        df =load_history ()
        last =df .iloc [-1 ].to_dict ()
        return last ,"parquet"

def run_forecasts (live ,models ,features ):
    now =datetime .now (timezone .utc )
    live .setdefault ("hour",now .hour )
    live .setdefault ("day",now .day )
    live .setdefault ("month",now .month )
    live .setdefault ("year",now .year )
    live .setdefault ("day_of_week",now .weekday ())
    live .setdefault ("sin_hour",np .sin (2 *np .pi *now .hour /24 ))
    live .setdefault ("cos_hour",np .cos (2 *np .pi *now .hour /24 ))
    live .setdefault ("sin_month",np .sin (2 *np .pi *now .month /12 ))
    live .setdefault ("cos_month",np .cos (2 *np .pi *now .month /12 ))
    live .setdefault ("sin_day",np .sin (2 *np .pi *now .day /31 ))
    live .setdefault ("cos_day",np .cos (2 *np .pi *now .day /31 ))
    live .setdefault ("sin_dow",np .sin (2 *np .pi *now .weekday ()/7 ))
    live .setdefault ("cos_dow",np .cos (2 *np .pi *now .weekday ()/7 ))

    results ={}
    for h ,model in models .items ():
        feats =features [h ]
        row ={f :live .get (f ,0.0 )for f in feats }
        X =pd .DataFrame ([row ])[feats ]
        pred =float (model .predict (X )[0 ])
        results [h ]=max (0 ,round (pred ,1 ))
    return results ,now 




with st .sidebar :
    st .markdown ("""
    <div style="padding: 8px 0 24px 0;">
        <div style="font-family: 'Space Mono', monospace; font-size: 11px;
                    letter-spacing: 3px; color: #f59e0b; margin-bottom: 4px;">
            LAHORE
        </div>
        <div style="font-family: 'Space Mono', monospace; font-size: 18px;
                    font-weight: 700; color: #e8eaf2; line-height: 1.2;">
            AQI<br>PREDICTOR
        </div>
        <div style="font-size: 12px; color: #6b7a99; margin-top: 8px;">
            ML-powered air quality forecasting
        </div>
    </div>
    """,unsafe_allow_html =True )

    st .divider ()

    page =st .radio (
    "Navigation",
    ["  Live Dashboard","  Historical EDA","  Model Insights","  Data Explorer"],
    label_visibility ="collapsed"
    )

    st .divider ()

    st .markdown ("""
    <div style="font-size: 11px; color: #6b7a99; line-height: 1.8;">
        <div style="font-family: 'Space Mono',monospace; color: #f59e0b; 
                    font-size: 10px; letter-spacing: 2px; margin-bottom: 8px;">
            STATION INFO
        </div>
         Lahore, Punjab<br>
         31.55°N, 74.34°E<br>
         Station: lahore_main<br>
         Source: AQICN + Open-Meteo
    </div>
    """,unsafe_allow_html =True )

    st .divider ()

    if st .button ("  Refresh Data",use_container_width =True ):
        st .cache_data .clear ()
        st .rerun ()




df =load_history ()
models ,feat_sets =load_models ()




if "Live Dashboard"in page :


    st .markdown ("""
    <div style="display:flex; align-items:baseline; gap:16px; margin-bottom:4px;">
        <span style="font-family:'Space Mono',monospace; font-size:26px; 
                     font-weight:700; color:#e8eaf2;">Live AQI Status</span>
        <span style="font-family:'Space Mono',monospace; font-size:11px; 
                     letter-spacing:2px; color:#f59e0b;">LAHORE</span>
    </div>
    """,unsafe_allow_html =True )

    with st .spinner ("Fetching live data..."):
        live ,source =get_live_features ()
        forecasts ,now =run_forecasts (live ,models ,feat_sets )

    current_aqi =live .get ("aqi")
    src_badge =" Online Store"if source =="online"else " Parquet Cache"


    if current_aqi and current_aqi >150 :
        color ="#ef4444"if current_aqi >200 else "#f97316"
        st .markdown (f"""
        <div class="alert-banner" style="background:rgba({
        '239,68,68'if current_aqi >200 else '249,115,22'},0.1);
            border-color:{color }; color:{color };">
             HAZARD ALERT — AQI {current_aqi :.0f} detected.
            {"Dangerous for all groups. Avoid outdoor activity."
        if current_aqi >200 else "Unhealthy for sensitive groups. Limit exposure."}
        </div>
        """,unsafe_allow_html =True )


    col_aqi ,col_meta =st .columns ([1 ,2 ])

    with col_aqi :
        color =aqi_color (current_aqi )
        label =aqi_label (current_aqi )
        emoji =aqi_emoji (current_aqi )
        st .markdown (f"""
        <div class="aqi-card" style="border-color:{color }40; min-height:180px;
             background: linear-gradient(135deg, #111520, {color }10);">
            <div class="metric-label">CURRENT AQI</div>
            <div class="metric-value" style="color:{color }; font-size:64px;">
                {f"{current_aqi :.0f}"if current_aqi is not None else "N/A"}
            </div>
            <div style="font-size:18px; margin:8px 0;">{emoji } {label }</div>
            <div class="metric-sub">{now .strftime ('%H:%M UTC')} · {src_badge }</div>
        </div>
        """,unsafe_allow_html =True )

    with col_meta :
        c1 ,c2 ,c3 =st .columns (3 )
        with c1 :
            st .metric ("PM2.5",f"{live .get ('pm25','N/A')} µg/m³")
            st .metric ("PM10",f"{live .get ('pm10','N/A')} µg/m³")
        with c2 :
            st .metric ("Temperature",f"{live .get ('temperature','N/A')} °C")
            st .metric ("Humidity",f"{live .get ('humidity','N/A')} %")
        with c3 :
            st .metric ("Wind Speed",f"{live .get ('wind_speed','N/A')} km/h")
            st .metric ("NO₂",f"{live .get ('no2','N/A')}")


    st .markdown ('<div class="section-header">3-DAY FORECAST</div>',unsafe_allow_html =True )

    f_cols =st .columns (len (forecasts ))
    for i ,(horizon ,pred )in enumerate (forecasts .items ()):
        hours =int (horizon .replace ("h",""))
        target_time =now +timedelta (hours =hours )
        color =aqi_color (pred )
        with f_cols [i ]:
            st .markdown (f"""
            <div class="aqi-card" style="border-color:{color }40;
                 background: linear-gradient(135deg, #111520, {color }08);">
                <div class="metric-label">+{horizon }</div>
                <div class="metric-value" style="color:{color }; font-size:42px;">
                    {pred :.0f}
                </div>
                <div style="font-size:14px; margin:4px 0;">{aqi_emoji (pred )} {aqi_label (pred )}</div>
                <div class="metric-sub">{target_time .strftime ('%a %d %b, %H:%M')}</div>
            </div>
            """,unsafe_allow_html =True )


    if forecasts :
        st .markdown ('<div class="section-header">FORECAST TRAJECTORY</div>',unsafe_allow_html =True )

        times =[now ]+[now +timedelta (hours =int (h .replace ("h","")))for h in forecasts ]
        values =[current_aqi or 0 ]+list (forecasts .values ())
        colors =[aqi_color (v )for v in values ]

        fig =go .Figure ()
        for i in range (len (times )-1 ):
            fig .add_trace (go .Scatter (
            x =[times [i ],times [i +1 ]],
            y =[values [i ],values [i +1 ]],
            mode ="lines",
            line =dict (color =colors [i +1 ],width =3 ),
            showlegend =False ,
            ))

        fig .add_trace (go .Scatter (
        x =times ,y =values ,
        mode ="markers+text",
        marker =dict (size =14 ,color =colors ,line =dict (width =2 ,color ="#0a0c10")),
        text =[f"AQI {v :.0f}"for v in values ],
        textposition ="top center",
        textfont =dict (family ="Space Mono",size =11 ,color ="#e8eaf2"),
        showlegend =False ,
        ))


        for thresh ,color ,label in [
        (50 ,"#10b981","Good"),(100 ,"#fbbf24","Moderate"),
        (150 ,"#f97316","Unhealthy S."),(200 ,"#ef4444","Unhealthy")
        ]:
            fig .add_hline (y =thresh ,line_dash ="dot",line_color =color ,
            opacity =0.4 ,annotation_text =label ,
            annotation_font_size =10 ,annotation_font_color =color )

        fig .update_layout (
        **plotly_dark (),
        height =300 ,
        xaxis_title =None ,
        yaxis_title ="AQI",
        title =dict (text ="Predicted AQI Trajectory",font =dict (family ="Space Mono",size =13 )),
        )
        st .plotly_chart (fig ,use_container_width =True )


    st .markdown ('<div class="section-header">RECENT 7-DAY HISTORY</div>',unsafe_allow_html =True )

    cutoff =df ["event_timestamp"].max ()-pd .Timedelta (days =7 )
    recent =df [df ["event_timestamp"]>=cutoff ]

    fig2 =go .Figure ()
    fig2 .add_trace (go .Scatter (
    x =recent ["event_timestamp"],y =recent ["aqi"],
    mode ="lines",
    line =dict (color ="#f59e0b",width =1.5 ),
    fill ="tozeroy",
    fillcolor ="rgba(245,158,11,0.08)",
    name ="AQI",
    ))
    for thresh ,color ,label in [
    (50 ,"#10b981","Good"),(100 ,"#fbbf24","Moderate"),
    (150 ,"#f97316","Unhealthy S."),(200 ,"#ef4444","Unhealthy")
    ]:
        fig2 .add_hline (y =thresh ,line_dash ="dot",line_color =color ,opacity =0.4 ,
        annotation_text =label ,annotation_font_size =9 ,
        annotation_font_color =color )

    fig2 .update_layout (
    **plotly_dark (),
    height =260 ,
    xaxis_title =None ,yaxis_title ="AQI",
    title =dict (text ="Hourly AQI — Last 7 Days",
    font =dict (family ="Space Mono",size =12 )),
    showlegend =False ,
    )
    st .plotly_chart (fig2 ,use_container_width =True )





elif "Historical EDA"in page :

    st .markdown ("""
    <div style="font-family:'Space Mono',monospace; font-size:22px;
                font-weight:700; margin-bottom:4px;">Historical Analysis</div>
    <div style="font-size:13px; color:#6b7a99; margin-bottom:24px;">
        4-year dataset · Jun 2022 – May 2026 · Lahore, Pakistan
    </div>
    """,unsafe_allow_html =True )


    def aqi_cat (v ):
        if v <=50 :return "Good"
        elif v <=100 :return "Moderate"
        elif v <=150 :return "Unhealthy (Sensitive)"
        elif v <=200 :return "Unhealthy"
        elif v <=300 :return "Very Unhealthy"
        else :return "Hazardous"

    df ["aqi_category"]=df ["aqi"].apply (aqi_cat )

    c1 ,c2 ,c3 ,c4 ,c5 =st .columns (5 )
    c1 .metric ("Total Hours",f"{len (df ):,}")
    c2 .metric ("Mean AQI",f"{df ['aqi'].mean ():.1f}")
    c3 .metric ("Median AQI",f"{df ['aqi'].median ():.1f}")
    c4 .metric ("Max AQI",f"{df ['aqi'].max ():.0f}")
    c5 .metric ("Hazardous %",f"{(df ['aqi']>300 ).mean ()*100 :.1f}%")

    tab1 ,tab2 ,tab3 ,tab4 =st .tabs (["📊 Distribution","📅 Seasonal"," Pollutants","🌡️ Weather"])

    with tab1 :
        col_a ,col_b =st .columns (2 )

        with col_a :
            fig =go .Figure ()
            fig .add_trace (go .Histogram (
            x =df ["aqi"],nbinsx =60 ,
            marker_color ="#f59e0b",opacity =0.8 ,name ="AQI",
            ))
            fig .add_vline (x =df ["aqi"].mean (),line_dash ="dash",
            line_color ="#ef4444",annotation_text =f"Mean {df ['aqi'].mean ():.0f}")
            fig .add_vline (x =df ["aqi"].median (),line_dash ="dash",
            line_color ="#10b981",annotation_text =f"Median {df ['aqi'].median ():.0f}")
            fig .update_layout (**plotly_dark (),height =320 ,
            title ="AQI Distribution",
            xaxis_title ="AQI",yaxis_title ="Count")
            st .plotly_chart (fig ,use_container_width =True )

        with col_b :
            cat_counts =df ["aqi_category"].value_counts ()
            cat_colors_map ={
            "Good":"#10b981","Moderate":"#fbbf24",
            "Unhealthy (Sensitive)":"#f97316","Unhealthy":"#ef4444",
            "Very Unhealthy":"#8b5cf6","Hazardous":"#991b1b",
            }
            fig =go .Figure (go .Pie (
            labels =cat_counts .index ,
            values =cat_counts .values ,
            hole =0.55 ,
            marker_colors =[cat_colors_map .get (c ,"#6b7a99")for c in cat_counts .index ],
            textfont =dict (family ="DM Sans",size =12 ),
            ))
            fig .update_layout (**plotly_dark (),height =320 ,
            title ="AQI Category Breakdown",
            showlegend =True ,
            legend =dict (font =dict (size =11 )))
            st .plotly_chart (fig ,use_container_width =True )

    with tab2 :
        col_a ,col_b =st .columns (2 )

        with col_a :

            daily =df .groupby (df ["event_timestamp"].dt .date )["aqi"].mean ().reset_index ()
            daily .columns =["date","aqi"]
            daily ["date"]=pd .to_datetime (daily ["date"])
            fig =go .Figure ()
            fig .add_trace (go .Scatter (
            x =daily ["date"],y =daily ["aqi"],
            mode ="lines",line =dict (color ="#f59e0b",width =1.2 ),
            fill ="tozeroy",fillcolor ="rgba(245,158,11,0.07)",
            ))
            for thresh ,color ,label in [(50 ,"#10b981","Good"),(100 ,"#fbbf24","Moderate"),
            (150 ,"#f97316","Unhealthy S."),(200 ,"#ef4444","Unhealthy")]:
                fig .add_hline (y =thresh ,line_dash ="dot",line_color =color ,opacity =0.5 ,
                annotation_text =label ,annotation_font_size =9 ,annotation_font_color =color )
            fig .update_layout (**plotly_dark (),height =300 ,
            title ="Daily Average AQI Over Time",xaxis_title =None ,yaxis_title ="AQI")
            st .plotly_chart (fig ,use_container_width =True )

        with col_b :

            month_names =["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            monthly =df .groupby (df ["event_timestamp"].dt .month )["aqi"].mean ()
            bar_colors =[aqi_color (v )for v in monthly .values ]
            fig =go .Figure (go .Bar (
            x =[month_names [i -1 ]for i in monthly .index ],
            y =monthly .values ,
            marker_color =bar_colors ,
            text =[f"{v :.0f}"for v in monthly .values ],
            textposition ="outside",
            textfont =dict (family ="Space Mono",size =10 ),
            ))
            fig .update_layout (**plotly_dark (),height =300 ,
            title ="Monthly Average AQI",xaxis_title =None ,yaxis_title ="AQI")
            st .plotly_chart (fig ,use_container_width =True )


        hourly =df .groupby ("hour")["aqi"].mean ()
        fig =go .Figure (go .Scatter (
        x =hourly .index ,y =hourly .values ,
        mode ="lines+markers",
        line =dict (color ="#8b5cf6",width =2.5 ),
        marker =dict (size =7 ,color =[aqi_color (v )for v in hourly .values ]),
        fill ="tozeroy",fillcolor ="rgba(139,92,246,0.06)",
        ))
        fig .update_layout (**plotly_dark (
        xaxis =dict (tickmode ="linear",dtick =2 ),
        height =260 ,
        title ="Average AQI by Hour of Day (UTC)",
        xaxis_title ="Hour (UTC)",yaxis_title ="AQI"))
        st .plotly_chart (fig ,use_container_width =True )

    with tab3 :
        polls =["pm25","pm10","no2","so2","co","o3"]
        poll_colors =["#f59e0b","#ef4444","#8b5cf6","#10b981","#06b6d4","#f97316"]
        poll_names =["PM2.5","PM10","NO₂","SO₂","CO","O₃"]

        cols =st .columns (2 )
        for i ,(poll ,color ,name )in enumerate (zip (polls ,poll_colors ,poll_names )):
            if poll not in df .columns :continue 
            with cols [i %2 ]:
                daily =df .groupby (df ["event_timestamp"].dt .date )[poll ].mean ().reset_index ()
                daily .columns =["date",poll ]
                daily ["date"]=pd .to_datetime (daily ["date"])
                fig =go .Figure (go .Scatter (
                x =daily ["date"],y =daily [poll ],
                mode ="lines",line =dict (color =color ,width =1.2 ),
                fill ="tozeroy",fillcolor =f"rgba({','.join (str (int (c ))for c in px .colors .hex_to_rgb (color ))},0.07)",
                ))
                fig .update_layout (**plotly_dark (
                height =220 ,
                title =f"{name } Over Time",
                xaxis_title =None ,yaxis_title =name ,
                margin =dict (l =20 ,r =20 ,t =36 ,b =20 )))
                st .plotly_chart (fig ,use_container_width =True )

    with tab4 :
        weather_feats =["temperature","humidity","wind_speed","cloud_cover","precipitation","boundary"]
        weather_names =["Temperature (°C)","Humidity (%)","Wind Speed (km/h)","Cloud Cover (%)","Precipitation","Boundary Layer"]
        w_colors =["#ef4444","#06b6d4","#10b981","#6b7a99","#8b5cf6","#f97316"]

        cols =st .columns (2 )
        for i ,(feat ,name ,color )in enumerate (zip (weather_feats ,weather_names ,w_colors )):
            if feat not in df .columns :continue 
            with cols [i %2 ]:
                fig =go .Figure (go .Scatter (
                x =df [feat ],y =df ["aqi"],
                mode ="markers",
                marker =dict (size =2 ,color =color ,opacity =0.3 ),
                ))
                fig .update_layout (**plotly_dark (
                height =220 ,
                title =f"AQI vs {name }",
                xaxis_title =name ,yaxis_title ="AQI",
                margin =dict (l =20 ,r =20 ,t =36 ,b =20 )))
                st .plotly_chart (fig ,use_container_width =True )





elif "Model Insights"in page :

    st .markdown ("""
    <div style="font-family:'Space Mono',monospace; font-size:22px;
                font-weight:700; margin-bottom:4px;">Model Insights</div>
    <div style="font-size:13px; color:#6b7a99; margin-bottom:24px;">
        SHAP feature importance · 3 trained forecasting horizons
    </div>
    """,unsafe_allow_html =True )


    horizon_labels ={"24h":"Next 24 Hours","48h":"Next 48 Hours","72h":"Next 72 Hours"}

    cols =st .columns (3 )
    for i ,(h ,label )in enumerate (horizon_labels .items ()):
        mp =MODELS_DIR /f"best_model_{h }.pkl"
        fp =MODELS_DIR /f"features_{h }.pkl"
        exists =mp .exists ()and fp .exists ()

        if exists :
            try :
                import joblib as _jl 
                _m =_jl .load (mp )
                model_name =type (_m ).__name__ 
            except Exception :
                model_name ="Unknown"
        else :
            model_name ="Not loaded"
        with cols [i ]:
            color ="#10b981"if exists else "#ef4444"
            st .markdown (f"""
            <div class="aqi-card" style="border-color:{color }30; text-align:left;">
                <div class="metric-label">HORIZON</div>
                <div style="font-family:'Space Mono',monospace; font-size:20px;
                            font-weight:700; color:{color };">+{h }</div>
                <div style="font-size:13px; margin:8px 0; color:#e8eaf2;">{label }</div>
                <div style="font-size:11px; color:#6b7a99;">Model: {model_name }</div>
                <div style="font-size:11px; color:#6b7a99; margin-top:4px;">
                    {' Loaded'if exists else ' Not found'}
                </div>
            </div>
            """,unsafe_allow_html =True )


    st .markdown ('<div class="section-header">SHAP FEATURE IMPORTANCE</div>',unsafe_allow_html =True )

    shap_tab1 ,shap_tab2 ,shap_tab3 =st .tabs (["24h Model","48h Model","72h Model"])
    for tab ,h in zip ([shap_tab1 ,shap_tab2 ,shap_tab3 ],["24h","48h","72h"]):
        with tab :
            shap_path =MODELS_DIR /f"shap_{h }.png"
            if shap_path .exists ():
                st .image (str (shap_path ),caption =f"SHAP Feature Importance — {h } Forecast",
                use_container_width =True )
            else :
                st .info (f"SHAP plot not found for {h }. Run model_training.py to generate.")


    st .markdown ('<div class="section-header">FEATURE CORRELATION MATRIX</div>',unsafe_allow_html =True )

    corr_cols =["aqi","pm25","pm10","no2","so2","co","o3",
    "temperature","humidity","wind_speed","boundary",
    "aqi_lag1","aqi_lag24","pm25_roll_mean_24","aqi_roll_mean_24"]
    corr_cols =[c for c in corr_cols if c in df .columns ]
    corr =df [corr_cols ].corr ()

    fig =go .Figure (go .Heatmap (
    z =corr .values ,
    x =corr .columns ,
    y =corr .columns ,
    colorscale ="RdBu_r",
    zmid =0 ,
    text =np .round (corr .values ,2 ),
    texttemplate ="%{text}",
    textfont =dict (size =9 ),
    colorbar =dict (tickfont =dict (size =10 )),
    ))
    fig .update_layout (**plotly_dark (
    height =520 ,
    title ="Feature Correlation Matrix",
    xaxis =dict (tickfont =dict (size =10 )),
    yaxis =dict (tickfont =dict (size =10 ),autorange ="reversed"),
    ))
    st .plotly_chart (fig ,use_container_width =True )


    st .markdown ('<div class="section-header">TOP CORRELATIONS WITH AQI</div>',unsafe_allow_html =True )

    aqi_corr =df [corr_cols ].corr ()["aqi"].drop ("aqi").sort_values ()
    bar_colors =["#ef4444"if v <0 else "#10b981"for v in aqi_corr .values ]

    fig =go .Figure (go .Bar (
    y =aqi_corr .index ,
    x =aqi_corr .values ,
    orientation ="h",
    marker_color =bar_colors ,
    text =[f"{v :.3f}"for v in aqi_corr .values ],
    textposition ="outside",
    textfont =dict (family ="Space Mono",size =10 ),
    ))
    fig .update_layout (**plotly_dark (),height =450 ,
    xaxis_title ="Pearson Correlation",yaxis_title =None ,
    title ="Feature Correlation with AQI")
    st .plotly_chart (fig ,use_container_width =True )





elif "Data Explorer"in page :

    st .markdown ("""
    <div style="font-family:'Space Mono',monospace; font-size:22px;
                font-weight:700; margin-bottom:4px;">Data Explorer</div>
    <div style="font-size:13px; color:#6b7a99; margin-bottom:24px;">
        Browse and filter the feature dataset
    </div>
    """,unsafe_allow_html =True )

    col1 ,col2 ,col3 =st .columns (3 )
    with col1 :
        date_min =df ["event_timestamp"].min ().date ()
        date_max =df ["event_timestamp"].max ().date ()
        date_range =st .date_input ("Date Range",
        value =(date_max -pd .Timedelta (days =30 ),date_max ),
        min_value =date_min ,max_value =date_max )

    with col2 :
        aqi_min ,aqi_max =st .slider ("AQI Range",0 ,350 ,(0 ,350 ))

    with col3 :
        col_opts =["aqi","pm25","pm10","no2","so2","co","o3",
        "temperature","humidity","wind_speed","boundary"]
        col_opts =[c for c in col_opts if c in df .columns ]
        selected_col =st .selectbox ("Plot Column",col_opts )


    mask =(
    (df ["event_timestamp"].dt .date >=date_range [0 ])&
    (df ["event_timestamp"].dt .date <=date_range [1 ])&
    (df ["aqi"]>=aqi_min )&(df ["aqi"]<=aqi_max )
    )if len (date_range )==2 else pd .Series (True ,index =df .index )

    filtered =df [mask ].copy ()
    st .caption (f"Showing {len (filtered ):,} rows")


    fig =go .Figure (go .Scatter (
    x =filtered ["event_timestamp"],y =filtered [selected_col ],
    mode ="lines",
    line =dict (color ="#f59e0b",width =1.2 ),
    fill ="tozeroy",fillcolor ="rgba(245,158,11,0.07)",
    ))
    fig .update_layout (**plotly_dark (),height =280 ,
    title =f"{selected_col .upper ()} over selected period",
    xaxis_title =None ,yaxis_title =selected_col )
    st .plotly_chart (fig ,use_container_width =True )


    display_cols =["event_timestamp","aqi","pm25","pm10","temperature",
    "humidity","wind_speed","no2","so2"]
    display_cols =[c for c in display_cols if c in filtered .columns ]

    st .dataframe (
    filtered [display_cols ].tail (200 ).sort_values ("event_timestamp",ascending =False ),
    use_container_width =True ,height =380 ,
    )


    csv =filtered [display_cols ].to_csv (index =False )
    st .download_button (
    "  Download Filtered CSV",
    data =csv ,
    file_name =f"lahore_aqi_{date_range [0 ]}_{date_range [1 ]}.csv",
    mime ="text/csv",
    )