import numpy as np
import plotly.graph_objects as go
import base64

from config import STRIKE_ZONE, MOVEMENT_SCALE, MOVEMENT_THRESHOLD


def load_svg(svg_path):
    try:
        with open(svg_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None

def create_viz(plate_x, plate_z, batter_handedness, pfx_x=0, pfx_z=0, release_speed=92, chart_height=600, pitcher_handedness='R'):
    fig = go.Figure()
    
    sz_left = STRIKE_ZONE['left']
    sz_right = STRIKE_ZONE['right']
    sz_bottom = STRIKE_ZONE['bottom']
    sz_top = STRIKE_ZONE['top']
    
    # strike zone
    fig.add_shape(
        type="rect",
        x0=sz_left, y0=sz_bottom, x1=sz_right, y1=sz_top,
        line=dict(color="red", width=2),
        fillcolor="rgba(255,0,0,0.05)"
    )
    
    # batter silhouette 
    if batter_handedness == 'R': 
        batter_x = -2.6
        svg_file = "assets/svg/batter_right.svg"
    else:
        batter_x = 2.6
        svg_file = "assets/svg/batter_left.svg"
    
    svg_content = load_svg(svg_file)
    if svg_content:
        svg_b64 = base64.b64encode(svg_content.encode()).decode()
        svg_uri = f"data:image/svg+xml;base64,{svg_b64}"
        sz_height = sz_top - sz_bottom
        sz_width = sz_right - sz_left
        
        batter_height = sz_height * 2.7
        batter_width = sz_width * 6
        batter_y = sz_top + (batter_height - sz_height) * 0.55
        
        fig.add_layout_image(
            dict(
                source=svg_uri,
                xref="x",
                yref="y",
                x=batter_x,
                y=batter_y,
                sizex=batter_width,
                sizey=batter_height,
                xanchor="center",
                yanchor="top",
                opacity=1.0,
                layer="below"
            )
        )

    # ball
    in_zone = (sz_left <= plate_x <= sz_right) and (sz_bottom <= plate_z <= sz_top)
    ball_color = "red" if in_zone else "green"
    ball_edge_color = "white"
    
    # glow that intensifies with velocity
    glow_intensity = (release_speed - 70) / 35  # normalize
    log_intensity = np.log(1 + glow_intensity * 9) / np.log(10)
    glow_size = 30 + (log_intensity * 5)
    base_opacity = 0.01 + (log_intensity * 0.3)
    
    # gradient glow 
    for i, scale in enumerate([1.0, 0.75, 0.5]):
        fig.add_trace(go.Scatter(
            x=[plate_x],
            y=[plate_z],
            mode='markers',
            marker=dict(
                size=glow_size * scale,
                color='red',
                opacity=base_opacity * (1.5 - i * 0.3),
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.add_trace(go.Scatter(
        x=[plate_x],
        y=[plate_z],
        mode='markers',
        marker=dict(
            size=20,
            color=ball_color,
            line=dict(color=ball_edge_color, width=3),
            symbol='circle'
        ),
        name='Location',
        showlegend=False,
        hovertemplate=f"""<b>Ball Location</b><br>X: {plate_x:.2f} ft<br>Z: {plate_z:.2f} ft<br>{'Strike' if in_zone else 'Ball'}<br><br><b>Movement</b><br>H: {pfx_x:+.2f} ft<br>V: {pfx_z:+.2f} ft<br>Speed: {release_speed:.0f} mph<extra></extra>"""
    ))
    
    # movement vectors
    pfx_x_ft = pfx_x * MOVEMENT_SCALE
    pfx_z_ft = pfx_z * MOVEMENT_SCALE
    
    if abs(pfx_x) > MOVEMENT_THRESHOLD or abs(pfx_z) > MOVEMENT_THRESHOLD:
        fig.add_annotation(
            x=plate_x, y=plate_z,
            ax=plate_x - pfx_x_ft, ay=plate_z - pfx_z_ft,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=1, arrowsize=1,
            arrowwidth=2, arrowcolor='rgba(255,255,255,0.8)',
            text=""
        )
    
    # pitcher silhouette
    pitcher_svg_file = f"assets/svg/pitcher_{'right' if pitcher_handedness == 'R' else 'left'}.svg"
    pitcher_svg = load_svg(pitcher_svg_file)
    if pitcher_svg:
        pitcher_b64 = base64.b64encode(pitcher_svg.encode()).decode()
        pitcher_uri = f"data:image/svg+xml;base64,{pitcher_b64}"

        pitcher_width = 0.9
        pitcher_height = 1.1

        fig.add_layout_image(
            dict(
                source=pitcher_uri,
                xref="x",
                yref="y",
                x=0,
                y=sz_top + 1.5,
                sizex=pitcher_width*2,
                sizey=pitcher_height*2,
                xanchor="center",
                yanchor="top",
                opacity=1.0,
                layer="below"
            )
        )

    # home plate SVG  
    homeplate_svg = load_svg("assets/svg/homeplate.svg")
    if homeplate_svg:
        homeplate_b64 = base64.b64encode(homeplate_svg.encode()).decode()
        homeplate_uri = f"data:image/svg+xml;base64,{homeplate_b64}"
        
        plate_width = 2
        plate_depth = 0.85
        
        fig.add_layout_image(
            dict(
                source=homeplate_uri,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=plate_width,
                sizey=plate_depth,
                xanchor="center",
                yanchor="bottom",
                opacity=1.0,
                layer="below"
            )
        )
    
    fig.update_layout(
        xaxis=dict(
            range=[-4, 4],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.3)',
            title="",
            showticklabels=False,
            fixedrange=True 
        ),
        yaxis=dict(
            range=[-1, 5.5],
            showgrid=True,
            gridcolor='rgba(128, 128,128,0.2)',
            zeroline=False,
            title="",
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True
        ),
        plot_bgcolor='rgba(96, 147, 93,0.7)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=chart_height,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig
