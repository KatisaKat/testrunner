import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc, Input, Output
import plotly.express as px

df_comp = pd.read_csv("cleaned_data/cleaned_companies.csv")
df_di = pd.read_csv("cleaned_data/cleaned_dealInvestor.csv")
df_deals = pd.read_csv("cleaned_data/cleaned_deals.csv") 
df_invs = pd.read_csv("cleaned_data/cleaned_investor.csv")
df_eco = pd.read_csv("cleaned_data/cleaned_ecosystem.csv")   

#------------------------------------ INVESTMENTS OVER TIME ------------------------------------
# Group by year and sum the investment amount
investment_trends = df_deals.groupby('year')['amount'].sum().reset_index()
fig_investment = px.line(investment_trends, x='year', y='amount', markers=True,
                         title='Total Investment in Canadian Startups (2019-2024)')

# Count number of deals per year
deal_volume = df_deals.groupby('year')['id'].count().reset_index()
deal_volume.rename(columns={'id': 'num_deals'}, inplace=True)
fig_deals = px.bar(deal_volume, x='year', y='num_deals',
                   title='Number of Investment Deals Per Year (2019-2024)', color='num_deals')

# Define deal size categories
def categorize_deal(amount):
    if amount < 100000:
        return '<$100K'
    elif 1000000 <= amount < 5000000:
        return '$1M-$5M'
    elif amount >= 100000000:
        return '$100M+'
    else:
        return 'Other'

df_deals['deal_size_category'] = df_deals['amount'].apply(categorize_deal)

deal_size_trends = df_deals.groupby(['year', 'deal_size_category'])['amount'].sum().reset_index()
deal_size_pivot = deal_size_trends.pivot(index='year', columns='deal_size_category', values='amount')
fig_deal_size = px.bar(deal_size_trends, x='year', y='amount', color='deal_size_category',
                        title='Investment Distribution by Deal Size (2019-2024)', barmode='stack')

# Merge deals with ecosystems
df_deals_regions = df_deals.merge(df_eco, on='ecosystemName', how='left')
region_trends = df_deals_regions.groupby(['year', 'province'])['amount'].sum().reset_index()
fig_region = px.line(region_trends, x='year', y='amount', color='province', markers=True,
                      title='Investment Trends by Region (2019-2024)')

# Data cleaning
valid_deals = df_deals_regions [(df_deals_regions ['roundType'] != 'Unknown') & (~df_deals_regions['province'].isna())]
valid_deals['date'] = pd.to_datetime(valid_deals['date'])
valid_deals['year'] = valid_deals['date'].dt.year
valid_deals = valid_deals[valid_deals['year'] != 2025]


# Get dropdown options
year_options = sorted(valid_deals['year'].unique())
province_options = sorted(valid_deals['province'].dropna().unique())
stage_options = sorted(valid_deals['roundType'].unique())
stage_options = [stage for stage in stage_options if stage.lower() != 'unknown']

#------------------------------------ FUNDING STAGES  ------------------------------------

def clean_stage(stage):
    mapping = {
        "pre seed": "Pre-Seed",
        "seed": "Seed",
        "series a": "Series A",
        "series b": "Series B",
        "series c": "Series C",
        "series d": "Series D+",
        "series e": "Series D+",
        "series f": "Series D+",
        "growth": "Growth",
        "ipo": "IPO"
    }
    return mapping.get(stage.lower(), "Other")

df_deals["roundType"] = df_deals["roundType"].astype(str).apply(clean_stage)

# Proportion of Deals per Investment Stage
# Count the number of deals at each funding stage
stage_counts = df_deals['roundType'].value_counts(normalize=True) * 100  # Convert to percentage
stage_counts = stage_counts.reset_index()  # Reset index to turn it into a DataFrame
stage_counts.columns = ['Funding Stage', 'Proportion']  # Rename columns for clarity

# Create a pie chart
fig_stage_proportion = px.pie(stage_counts, 
                               names='Funding Stage', 
                               values='Proportion',
                               title='Proportion of Deals at Each Investment Stage',
                               color_discrete_sequence=px.colors.qualitative.Prism)

# Average Deal Size per Stage Over Time
deal_size = df_deals.groupby(["year", "roundType"])['amount'].mean().reset_index()
fig_avg_deal_size = px.line(deal_size, x='year', y='amount', color='roundType',
                            title='Average Deal Size Per Stage Over Time',
                            labels={'amount': 'Average Deal Size ($)', 'year': 'Year'})

# Trends in Number and Size of Deals per Stage Over Years
deal_trends = df_deals.groupby(["year", "roundType"]).agg({"id": "count", "amount": "sum"}).reset_index()
deal_trends.rename(columns={"id": "num_deals"}, inplace=True)
fig_deal_trends = px.bar(deal_trends, x='year', y='num_deals', color='roundType',
                         title='Trends in Number of Deals Per Stage Over Years',
                         labels={'num_deals': 'Number of Deals', 'year': 'Year'},
                         barmode='stack', color_discrete_sequence=px.colors.qualitative.Dark24)

#------------------------------------ INVESTOR BEHAVIOUR ------------------------------------

# Categorize countries: Keep US & Canada, group others as 'Other International'
df_invs["country_grouped"] = df_invs["country"].apply(
    lambda x: x if x == "usa" else ("canada" if x == "canada" else "other International")
)

# Count number of investment firms by country group
investor_counts = df_invs.groupby("country_grouped").size().reset_index(name="count")

# Visualize Investment Firm Demographics with Plotly
fig1 = px.bar(
    investor_counts,
    x="country_grouped",
    y="count",
    title="Investment Firm Demographics",
    labels={"country_grouped": "Country Group", "count": "Number of Investment Firms"},
    color="country_grouped",  # Optional: color by country group
    color_discrete_sequence=px.colors.sequential.Viridis  # Optional: use 'Viridis' color palette
)

# Assuming the dataframe 'df_invs' is already defined
investment_counts = df_invs.explode("stages").groupby(["country", "stages"]).size().reset_index(name="count")

# Filtering out insignificant data (threshold of at least 5 firms per category)
investment_counts = investment_counts[investment_counts["count"] >= 5]

# Pivoting for better visualization
investment_pivot = investment_counts.pivot(index="country", columns="stages", values="count").fillna(0)

# Plot the heatmap using Plotly
fig2 = px.imshow(
    investment_pivot,
    title="Number of Significant Investments per Funding Stage by Country",
    labels={"x": "Funding Stage", "y": "Country", "color": "Number of Firms"},
    color_continuous_scale="Blues",  # Optional: change color scale
)

# Adjust layout for better presentation
fig2.update_xaxes(tickangle=45, tickmode='array')
fig2.update_layout(
    xaxis_title="Funding Stage",
    yaxis_title="Country",
    xaxis=dict(tickmode='array'),
    yaxis=dict(tickmode='array'),
    coloraxis_colorbar=dict(title="Number of Firms")
)

# Assuming the dataframe 'df_deals' is already defined
deal_size_by_stage = df_deals.groupby("roundType")["amount"].mean().reset_index(name="avg_deal_size")

# Plot the bar chart using Plotly with Cividis color palette
fig3 = px.bar(
    deal_size_by_stage,
    x="roundType",
    y="avg_deal_size",
    title="Average Deal Size Per Funding Stage",
    labels={"roundType": "Funding Stage", "avg_deal_size": "Average Deal Size"},
    color="roundType",  # Optional: color by funding stage for better visualization
    color_continuous_scale="Cividis"  # Set color scale to Cividis
)

# Assuming the dataframe 'df_di' and 'df_deals' are already defined
# Identify leading investors per stage
lead_investors_per_stage = df_di.groupby(["roundType", "investorName"]).agg(
    lead_count=("leadInvestorFlag", "sum"),
    total_deals=("dealId", "count")
).reset_index()

# Sort and select top investors per stage
top_lead_investors = lead_investors_per_stage.sort_values(
    by=["roundType", "lead_count", "total_deals"], ascending=[True, False, False]
).groupby("roundType").head(3)  # Top 3 investors per stage

# Influence on funding success
funding_data = df_deals.merge(df_di, on="id", how="left")
investor_funding = funding_data.groupby("investorName").agg(
    total_funding=("amount", "sum"),
    avg_funding=("amount", "mean"),
    total_deals=("id", "count")
).reset_index()

# Filter for leading investors
top_investors = investor_funding[investor_funding["investorName"].isin(top_lead_investors["investorName"])]

# Plot using Plotly
fig4 = px.bar(
    top_investors.sort_values("total_funding", ascending=False),
    x="investorName",
    y="total_funding",
    color="total_deals",
    title="Top Lead Investors and their Funding Influence",
    labels={"investorName": "Investor Name", "total_funding": "Total Funding Received"},
    color_continuous_scale="Inferno_r"
)

# Adjust layout for better readability
fig4.update_layout(
    xaxis_title="Investor Name",
    yaxis_title="Total Funding Received",
    xaxis_tickangle=45,
    legend_title="Total Deals"
)

# ------------------------------------ REGIONAL INSIGHTS ------------------------------------
# top investment categories
cats = df_deals.groupby('primaryTag').agg({'amount': 'sum'}).reset_index()
cats = cats.sort_values(by='amount', ascending=False)
cats = cats.head(10) # top 10
top10_cats = px.bar(cats,
             x='amount', y='primaryTag', 
             color = 'primaryTag',
             title='Top 10 Investment Categories by Amount',
             labels={'primaryTag': 'Investment Category', 'amount': 'Total Investment ($)'},
             color_discrete_sequence=px.colors.qualitative.Prism,)
top10_cats.update_layout(showlegend=False)  

# average deal size
avg_deal = df_deals.groupby('ecosystemName')['amount'].mean().reset_index()
avg_reg = px.bar(avg_deal, x='ecosystemName', y='amount', 
             labels={'ecosystemName':'Region', 'amount':'Average Investment Volume'},
             title='Average Deal Size by Region',)
avg_reg.update_xaxes(categoryorder='total descending')

# total investment vol and category
cat_pref = df_deals.groupby(['ecosystemName', 'primaryTag'])['amount'].sum().reset_index()
cat_pref = cat_pref.sort_values(by='amount', ascending=False)
top20 = df_deals.groupby('primaryTag')['amount'].sum().sort_values(ascending=False)
top20 = top20.head(25).index
cat_pref = cat_pref[cat_pref['primaryTag'].isin(top20)]

inv_reg = px.bar(cat_pref, x='ecosystemName', y='amount', color='primaryTag',
             labels={'ecosystemName':'Region', 'amount':'Investment Volume', 'primaryTag':'Categories (Top 25)'},
             title='Category Preferences and Investment Volume by Region', barmode='stack',
             color_discrete_sequence=px.colors.qualitative.Dark24,)
inv_reg.update_xaxes(categoryorder='total descending')

# map visualisation by key headquarter regions
# longitude and latitude of major regions of interest
hq_loc = {
    'toronto': [43.6511, -79.3470],
    'montreal': [45.5019, -73.5674],
    'waterloo': [43.4643, -80.5204],
    'ottawa': [45.4235, -75.6979],
    'quebec': [46.8131, -71.2075],
    'vancouver': [49.2827, -123.1207],
    'calgary': [51.0447, -114.0719],
    'edmonton': [53.5461, -113.4937],
    'winnipeg': [49.8954, -97.1385],
}

df_loc = pd.DataFrame(hq_loc).T.reset_index()
df_loc.columns = ['headquarters', 'lat', 'lon']

# grabs a subset of deals with relevant headquarters, finds its total investment vol and top inv categories
hq_trends = df_deals.groupby(['headquarters', 'primaryTag'])['amount'].sum().reset_index()

# finds top 3 categories for each major headquarter location
top_cats = []
for hq in hq_trends['headquarters']:
        hq_data = hq_trends[hq_trends['headquarters'] == hq]
        top4 = hq_data.sort_values(by='amount', ascending=False).head(4)
        top_cats.append(top4)
top4_df = pd.concat(top_cats, ignore_index=True)

hq_trends = top4_df.merge(df_loc, on='headquarters', how='left')
num_deals = df_deals.groupby('headquarters').size().reset_index(name='num_deals') # counts number of deals for each hq
hq_trends = hq_trends.merge(num_deals, on='headquarters', how='left')
hq_trends = hq_trends.dropna(subset=['lat', 'lon']) # drops irrelevant headquarter locations
hq_trends = hq_trends.sort_values(by="amount", ascending=False)

map_hq = px.scatter_map(
    hq_trends,
    lat='lat',
    lon='lon',
    size='amount',  # bubble size represents investment volume
    color="primaryTag", # color represents top category
    hover_name='headquarters',
    hover_data={'primaryTag': True, "amount": True, 'num_deals': True, "lat": False, "lon":False},
    labels={'amount':'Investment Volume by Category', 'primaryTag':'Top Investment Categories',
            'num_deals': 'Total Number of Deals for Headquarter'},
    color_discrete_sequence=px.colors.qualitative.Prism,
    size_max=60, 
    zoom=3
)

map_hq.update_layout(
    mapbox_style='carto-positron',
    mapbox_center={"lat": 56, "lon": -106}, 
)

# ------------------------- Putting everything into a dashboard -------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(id="everything",children=[
    html.H1("Canadian Tech Investments Dashboard", id="main-header"),
        dcc.Tabs([
            dcc.Tab(label='Investment Trends Over Time', className="section-header", children=[
                html.Div([
                    html.H3("Investment Trends Over Time", style={'text-align': 'center','margin':'50px 0px 75px 0px'}),
                    
                    dbc.Row(className="graph", children=[
                        dbc.Col([
                            html.Label("Select Year"),
                            dcc.Dropdown(
                                id='year-dropdown', 
                                options=[{'label': str(y), 'value': y} for y in year_options],
                                multi=True, value=[year_options[-1]]
                            ),
                            
                            html.Label("Select Province", style={'margin-top': '20px'}),
                            dcc.Dropdown(
                                id='province-dropdown', 
                                options=[{'label': p, 'value': p} for p in province_options],
                                multi=True, value=province_options
                            ),
                            
                            html.Label("Select Funding Stage", style={'margin-top': '20px'}),
                            dcc.Dropdown(
                                id='stage-dropdown', 
                                options=[{'label': s, 'value': s} for s in stage_options],
                                multi=True, value=stage_options
                            )
                        ], width=3, style={'padding': '20px'}),
                        dbc.Col([
                            dcc.Graph(
                                id='investment-bar-chart', 
                                style={'width': '100%', 'height': '600px'}
                            )
                        ], width=9)  # Graph takes more space
                    ], justify='start', style={'max-width': '90%', 'margin': 'auto'}),
                    html.P(className="writing hidden",children=[
                        html.B("Total Investment Growth: "),
                        "The Canadian tech sector saw an 82.47% increase in total investment, reflecting strong growth.", html.Br(),
                        html.B("Deal Volume & Funding Size: "),
                        "The number of deals dropped by 54.40%, but funding per deal increased, indicating a shift toward larger investments in fewer companies.", html.Br(),
                        html.B("Investment Size Trends: "),
                        "The $100M+ category dominated, with $24.9B invested, suggesting a preference for late-stage funding and fewer small investments.", html.Br(),
                        "Ontario led with $22 billion in investment, followed by British Columbia at $12 billion. Investment peaked in 2021–2022 for Ontario, BC, and Quebec, while other provinces saw no clear peak."
                    ], style={'margin-left': '50px', 'margin-top': '30px'}),
                ]),
                html.Div(className="graph-box hiddenleft",children=[
                    dcc.Tab(label='Total Investment', children=[dcc.Graph(figure=fig_investment, className="graph", style={'width': '100%', 'margin': 'auto'})]),
                    html.P([
                        html.B("Total Investment Growth"),
                        html.Br(),
                        "The Canadian tech sector experienced a significant surge in total investment, with an increase of 82.47%. This indicates that the sector has been flourishing and attracting more capital, showing a strong trend of growth, particularly in recent years.",
                    ], style={'margin-left': '75px', 'text-align': 'left'}),
                ]),
                html.Div(className="graph-box hiddenright",children=[
                    html.P([
                        html.B("Deal Volume & Funding Size"),
                        html.Br(),
                        "While the number of deals dropped by 54.40%, the average funding size per deal increased. This shift indicates a trend where investors are becoming more selective, focusing on fewer companies but making larger investments. The drop in deal volume might suggest that firms are prioritizing more substantial, possibly late-stage investments instead of spreading their capital across numerous early-stage ventures."
                    ], style={'margin-right': '75px', 'text-align': 'right'}),
                    dcc.Tab(label='Deal Volume', children=[dcc.Graph(figure=fig_deals,className="graph",style={'width': '80%', 'height': '500px', 'margin': 'auto'})]),
                ]),
                html.Div(className="graph-box hiddenleft",children=[
                    dcc.Tab(label='Deal Size Distribution', children=[dcc.Graph(figure=fig_deal_size,className="graph",style={'width': '80%', 'height': '500px', 'margin': 'auto'})]),
                    html.P([
                        html.B("Investment Size Trends"),
                        html.Br(),
                        "A significant portion of the investment, about $24.9 billion, went into the $100M+ category. This highlights a strong preference for late-stage investments, where larger sums are directed into more mature companies. The trend shows a reduction in smaller deals, emphasizing a shift towards funding well-established, high-growth companies rather than early-stage startups.",
                    ], style={'margin-left': '75px', 'text-align': 'left'}),
                ]),
                html.Div(className="graph-box hiddenright",children=[
                    html.P([
                            html.B("Ontario and Regional Investment Peaks"),
                            html.Br(),
                            "Ontario led the charge with $22 billion in investment, followed by British Columbia at $12 billion. Investment activity peaked in the years 2021–2022 for Ontario, BC, and Quebec, signaling the highest levels of capital influx during those years. In contrast, other provinces did not show a clear peak, reflecting potentially less consistent or lower overall investment activity.",
                        ], style={'margin-right': '75px', 'text-align': 'right'}),
                    dcc.Tab(label='Regional Trends', children=[dcc.Graph(figure=fig_region,className="graph",style={'width': '80%', 'height': '500px', 'margin': 'auto'})]),
                ]),
            ]),
            dcc.Tab(label=' Funding Stages Analysis', className="section-header", children=[
                html.H3("Funding Stages Analysis", style={'text-align': 'center','margin-top': '50px', 'margin-left': '50px'}),
                html.Div(className="graph-box",children=[
                    dcc.Tab(label='Proportion of Deals', children=[dcc.Graph(figure=fig_stage_proportion,className="graph")]),
                    html.P([
                        html.B("Deal Proportions by Stage"),
                        html.Br(),
                        "The Seed stage dominated, accounting for 36.34% of all deals. This was followed by Series A (15.61%) and Pre-seed (15.2%) deals, showing that early-stage investments still make up a significant portion of the overall deal volume. However, as the stages progress, the percentage of deals declines, indicating that there are fewer investments at later funding stages such as Series B, C, and beyond.",
                    ], style={'margin-left': '75px', 'text-align': 'left'}),
                ]),
                html.Div(className="graph-wrap",children=[
                    dcc.Tab(label='Average Deal Size Over Time', children=[dcc.Graph(figure=fig_avg_deal_size,className="graph",style={'flex': '1 1 40%','margin-right': '20px'})]),
                    dcc.Tab(label='Deal Trends Over Years', children=[dcc.Graph(figure=fig_deal_trends,className="graph",style={'flex': '1 1 40%','margin-left': '20px'})]),
                    html.P([
                        html.B("Average Deal Size"),
                        html.Br(),
                        "Deals in the Seed stage typically ranged between $3 million and $5 million, showing steady interest in early-stage ventures. In contrast, the Series D+ stage saw a dramatic rise, with deals peaking at $519.67 million in 2024, highlighting a trend toward significantly larger investments in more mature, late-stage companies.",
                    ], style={'flex': '1 1 25%', 'margin-top': '30px', 'margin-right': '30px', 'text-align': 'center'}),
                    html.P([
                        html.B("Trends Over Multiple Years"),
                        html.Br(),
                        "The number of Seed-stage deals peaked in 2021 with 253 deals, but by 2023, this number dropped to just 65, suggesting a reduction in early-stage investments. On the other hand, Series D+ investments grew substantially, even though the number of deals decreased, indicating a shift toward fewer but much larger investments in well-established companies.",
                    ], style={'flex': '1 1 25%', 'margin-top': '30px', 'margin-left': '30px', 'text-align': 'center'}),
                ]),
            ]),
            dcc.Tab(label='Investor Demographics & Behavior', className="section-header", children=[
                html.H3("Investor Demographics & Behavior", style={'text-align': 'center','margin-top': '50px', 'margin-left': '50px'}),
                html.Div(className="graph-box",children=[
                    dcc.Graph(
                        figure=fig1,
                        className="graph",
                        style={'width': '100%', 'margin': 'auto'}
                    ),
                    html.P([
                        html.B("Investor Demographics"),
                        html.Br(),
                        "The United States leads the investment landscape with 1096 firms, followed by Canada with 587 firms and other international investors at 580 firms. This shows the overwhelming dominance of US-based investment firms in the market, though Canada still plays a significant role in funding tech companies, with a notable number of firms actively participating.",
                    ], style={'margin-left': '75px', 'text-align': 'left'}),
                ]),
                html.Div(className="graph-box",children=[
                    html.P([
                        html.B("Investment Firms by Stage & Country"),
                        html.Br(),
                        "The US leads in investment activity across all funding stages, showing its dominance in supporting companies throughout their growth cycles. Canada is more active in earlier stages like Seed (125 firms), while other countries tend to have minimal involvement, particularly in the later stages, further reinforcing the US's lead."
                    ], style={'margin-right': '75px', 'text-align': 'right'}),
                    dcc.Graph(
                        figure=fig2,
                        className="graph",
                        style={'width': '100%', 'margin': 'auto'}
                    ),
                ]),
                html.Div(className="graph-box",children=[
                    dcc.Graph(
                        figure=fig3,
                        className="graph",
                        style={'width': '100%', 'margin': 'auto'}
                    ),
                    html.P([
                        html.B("Average Deal Size per Stage"),
                        html.Br(),
                        "Series F deals have an average deal size of $450 million, which is significantly larger than the Pre-Seed average of $698,000. This indicates that as companies mature through the funding stages, the investments grow substantially in size, reflecting the increasing maturity and valuation of the companies involved.",
                    ], style={'margin-left': '75px', 'text-align': 'left'}),
                ]),
                html.Div(className="graph-box",children=[
                    html.P([
                        html.B("Leading Investors"),
                        html.Br(),
                        "BDC Capital is a standout investor, leading with a total of $1.47 billion across 84 deals, marking a strong presence in the investment community. Other key players such as Techstars, Georgian, and Inovia Capital also have significant contributions, demonstrating their influence in funding across various stages.",
                    ], style={'margin-right': '75px', 'text-align': 'right'}),
                    dcc.Graph(
                        figure=fig4,
                        className="graph",
                        style={'width': '100%', 'margin': 'auto'}
                    ),
                ]),
                html.P([
                    html.B("Y Combinator and BDC Capital show consistent investment activity year over year. In contrast, other firms like Investissement Québec show more targeted investment activity, likely focused on specific sectors or regional trends. The fluctuation in activity levels underscores that some investors are more agile, reacting to market conditions, while others are more selective in their investment choices."),
                ], style={'font-size': '30px', 'margin': '35px 5% 100px 5%', 'text-align': 'left'}),
            ]),
            dcc.Tab(label='Regional Insights', className="section-header", children=[
                html.Div([
                    html.H3("Sectoral & Regional Insights", style={'text-align': 'center','margin-top': '50px', 'margin-left': '50px'}),
                    html.Div(className="graph-box",children=[
                        dcc.Graph(figure=top10_cats, className="graph",
                            style={'width': '100%', 'margin': 'auto'}),
                        html.P([
                            html.B("Top Investment Categories"),
                            html.Br(),
                            "The top sectors by total funding include FinTech ($7.4 billion), SaaS ($6.5 billion), and AI ($4.3 billion). These sectors continue to attract significant attention and investment, reflecting ongoing interest in technology that revolutionizes financial services, cloud-based solutions, and artificial intelligence-driven innovations.",
                        ], style={'margin-left': '75px', 'text-align': 'left'}),
                    ]),
                ]),
                html.Div([
                    html.H3("Investment Trends by Major Headquarter Locations",style={'text-align': 'center', 'margin-top': '50px', 'margin-left': '50px'}),
                    html.P(className="writing",children=[
                        "British Columbia leads with the highest average deal size of $23.5 million, primarily driven by investments in Cleantech and Biotech sectors. Waterloo follows closely with an average of $22.5 million, largely fueled by investments in AI and SaaS. In contrast, Toronto sees more deals overall, but the average deal size is lower at $18 million, with a focus on sectors like MarTech, FinTech, and HealthTech.",
                        html.Br(),
                        html.U("Hover over the headquarter to see information such as total investment funding for each category, and number of deals!"),
                    ], style={'margin-left': '50px', 'text-align': 'center'}),
                    dcc.Graph(
                        figure=map_hq, className="graph",
                        style={'margin': 'auto', 'width': '90%', 'height': '700px'}
                    )
                ]),
                html.Div(className="graph-wrap",children=[
                    dcc.Graph(figure=inv_reg, className="graph",
                            style={'flex': '1 1 25%', 'margin-top': '30px', 'margin-right': '30px', 'text-align': 'center'},
                    ),
                    dcc.Graph(figure=avg_reg, className="graph",
                            style={'flex': '1 1 25%', 'margin-top': '30px', 'margin-right': '30px', 'text-align': 'center'},
                    ),
                    html.P([
                        html.B("Leading Investors"),
                        html.Br(),
                        "Toronto dominates in terms of the number of deals but tends to have smaller deal sizes compared to British Columbia and Waterloo, which stand out for their larger investments. Quebec, Ottawa, and Alberta are emerging ecosystems that are beginning to attract more focused investments, particularly in specific sectors, although their deal sizes are typically smaller compared to the leading regions.",
                    ], style={'margin-top': '50px', 'margin-right': '75px', 'text-align': 'center'}),
                ]),
            ]),
        ]),
    html.Script(src="/assets/animation.js", defer=True),
])

port = int(os.environ.get("PORT", 8050))

# Run the app with Gunicorn on the assigned port
if __name__ == '__main__':
    # Gunicorn will handle the app server
    app.run_server(debug=True, host="0.0.0.0", port=port)

# Callbacks
@app.callback(
    Output('investment-bar-chart', 'figure'),
    Input('year-dropdown', 'value'),
    Input('province-dropdown', 'value'),
    Input('stage-dropdown', 'value')
)
def update_graph(selected_years, selected_provinces, selected_stages):
    filtered_df = valid_deals[
        (valid_deals['year'].isin(selected_years)) &
        (valid_deals['province'].isin(selected_provinces)) &
        (valid_deals['roundType'].isin(selected_stages))
    ]
    
    summary = filtered_df.groupby(['province'])['amount'].sum().reset_index()
    fig = px.bar(summary, x='province', y='amount', title='Investment Amount by Province',
                 labels={'province': 'Province', 'amount': 'Total Investment ($)'},
                 color='province', color_discrete_sequence=px.colors.qualitative.Prism,)
    fig.update_xaxes(categoryorder='total descending')
    return fig

if __name__ == '__main__':
    print("App is starting")
    app.run_server(debug=True)
