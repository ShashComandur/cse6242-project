import streamlit as st
import plotly.graph_objects as go


def display_prediction_results(results):
    outcome_cols = st.columns(4)
    with outcome_cols[0]:
        st.metric("Ball", f"{results.get('ball', 0) * 100:.1f}%")
    with outcome_cols[1]:
        st.metric("Strike", f"{results.get('strike', 0) * 100:.1f}%")
    with outcome_cols[2]:
        st.metric("Foul", f"{results.get('foul_ball', 0) * 100:.1f}%")
    with outcome_cols[3]:
        st.metric("In Play", f"{results.get('in_play', 0) * 100:.1f}%")
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(results.keys()),
            y=list(results.values()),
            marker=dict(
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                line=dict(color='white', width=2)
            ),
            text=[f"{v*100:.1f}%" for v in results.values()],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Probability Distribution",
            'font': {'size': 16, 'color': 'white'}
        },
        xaxis=dict(
            title="",
            tickfont=dict(size=12, color='white'),
            showgrid=False
        ),
        yaxis=dict(
            title="Probability",
            tickformat='.0%',
            range=[0, max(results.values()) * 1.15],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_batted_results(results):
    outcome_order = ['out', 'single', 'double', 'triple', 'home_run']
    ordered = {k: results.get(k, 0) for k in outcome_order if k in results}
    for k, v in results.items():
        if k not in ordered:
            ordered[k] = v

    labels = list(ordered.keys())
    values = list(ordered.values())
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    metric_cols = st.columns(len(labels))
    for col, label, value in zip(metric_cols, labels, values):
        with col:
            st.metric(label.replace('_', ' ').title(), f"{value * 100:.1f}%")

    fig = go.Figure(data=[go.Bar(
        x=[l.replace('_', ' ').title() for l in labels],
        y=values,
        marker=dict(color=colors[:len(labels)], line=dict(color='white', width=2)),
        text=[f"{v*100:.1f}%" for v in values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
    )])

    fig.update_layout(
        title={'text': "Batted Ball Outcomes", 'font': {'size': 16, 'color': 'white'}},
        xaxis=dict(title="", tickfont=dict(size=12, color='white'), showgrid=False),
        yaxis=dict(
            title="Probability",
            tickformat='.0%',
            range=[0, max(values) * 1.15],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
