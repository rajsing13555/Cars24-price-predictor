"""
build_dashboard.py
--------------------
Generates a self-contained, interactive BI dashboard (dashboard/dashboard.html)
that plays the role a Power BI dashboard would in this project: filterable
KPIs and charts a business/ops team could use to monitor the used-car
marketplace, built with Plotly.js so it opens directly in any browser with
zero server setup (double-click and it works).

Data is embedded directly in the HTML as JSON so the file is fully
portable (no fetch/CORS issues when opened locally).
"""

import pandas as pd
import json

df = pd.read_csv("data/car_listings.csv")
cols = ["brand", "model", "city", "car_age", "fuel_type", "transmission",
        "asking_price", "status", "days_to_sell", "sold_price", "listing_date"]
trim = df[cols].copy()
trim["listing_month"] = trim["listing_date"].str[:7]
trim = trim.drop(columns=["listing_date"])

records = trim.to_dict(orient="records")
data_json = json.dumps(records)

brands = sorted(trim["brand"].unique().tolist())
cities = sorted(trim["city"].unique().tolist())

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CARS24 Marketplace Intelligence Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --panel: #171a23; --panel-2: #1e222d; --border: #2a2f3d;
    --text: #e8e9ee; --text-dim: #9297a6; --accent: #4c7fee; --accent-2: #22c58b;
    --warn: #f0a63a; --danger: #e2504a;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  }}
  header {{
    padding: 20px 28px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
  }}
  header h1 {{ font-size: 19px; font-weight: 600; margin: 0; }}
  header p {{ font-size: 12.5px; color: var(--text-dim); margin: 2px 0 0; }}
  .filters {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  select {{
    background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 7px 10px; font-size: 13px; cursor: pointer;
  }}
  main {{ padding: 20px 28px 40px; max-width: 1400px; margin: 0 auto; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 20px; }}
  .kpi {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 18px;
  }}
  .kpi .label {{ font-size: 12px; color: var(--text-dim); margin-bottom: 6px; }}
  .kpi .value {{ font-size: 24px; font-weight: 600; }}
  .kpi .sub {{ font-size: 11.5px; color: var(--accent-2); margin-top: 4px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .grid .full {{ grid-column: 1 / -1; }}
  .chart-card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px 6px;
  }}
  .chart-card h3 {{ font-size: 13.5px; font-weight: 600; margin: 0 0 4px; color: var(--text); }}
  .chart-card p.desc {{ font-size: 11.5px; color: var(--text-dim); margin: 0 0 8px; }}
  @media (max-width: 900px) {{
    .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
    .grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<header>
  <div>
    <h1>CARS24 Marketplace Intelligence Dashboard</h1>
    <p>Used car listings &amp; sales performance — synthetic dataset, 8,000 listings across 10 cities</p>
  </div>
  <div class="filters">
    <select id="cityFilter"><option value="__all__">All cities</option></select>
    <select id="brandFilter"><option value="__all__">All brands</option></select>
  </div>
</header>

<main>
  <div class="kpi-row" id="kpiRow"></div>

  <div class="grid">
    <div class="chart-card">
      <h3>Monthly sold units &amp; average price</h3>
      <p class="desc">Volume and pricing trend over the last 18 months</p>
      <div id="chartTrend" style="height:300px;"></div>
    </div>
    <div class="chart-card">
      <h3>Average sold price by brand</h3>
      <p class="desc">Which brands command the highest resale value</p>
      <div id="chartBrand" style="height:300px;"></div>
    </div>
    <div class="chart-card">
      <h3>Sell-through rate &amp; avg days to sell by city</h3>
      <p class="desc">Ops metric: which cities move inventory fastest</p>
      <div id="chartCity" style="height:300px;"></div>
    </div>
    <div class="chart-card">
      <h3>Price spread by fuel type &amp; transmission</h3>
      <p class="desc">Distribution of sold price across configurations</p>
      <div id="chartFuel" style="height:300px;"></div>
    </div>
    <div class="chart-card full">
      <h3>Inventory ageing — active (unsold) listings</h3>
      <p class="desc">Listings still active, bucketed by days on platform — flags aging stock risk</p>
      <div id="chartAge" style="height:280px;"></div>
    </div>
  </div>
</main>

<script>
const RAW = {data_json};
const BRANDS = {json.dumps(brands)};
const CITIES = {json.dumps(cities)};

const cityFilter = document.getElementById('cityFilter');
const brandFilter = document.getElementById('brandFilter');
CITIES.forEach(c => cityFilter.innerHTML += `<option value="${{c}}">${{c}}</option>`);
BRANDS.forEach(b => brandFilter.innerHTML += `<option value="${{b}}">${{b}}</option>`);

const PLOT_LAYOUT_BASE = {{
  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: {{ color: '#e8e9ee', size: 11.5 }},
  margin: {{ l: 46, r: 16, t: 10, b: 40 }},
  xaxis: {{ gridcolor: '#2a2f3d', zerolinecolor: '#2a2f3d' }},
  yaxis: {{ gridcolor: '#2a2f3d', zerolinecolor: '#2a2f3d' }},
  legend: {{ orientation: 'h', y: 1.15 }},
}};

function fmtINR(n) {{
  if (n >= 100000) return '₹' + (n/100000).toFixed(1) + 'L';
  return '₹' + Math.round(n).toLocaleString('en-IN');
}}

function getFiltered() {{
  const c = cityFilter.value, b = brandFilter.value;
  return RAW.filter(r => (c === '__all__' || r.city === c) && (b === '__all__' || r.brand === b));
}}

function mean(arr) {{ return arr.length ? arr.reduce((a,b)=>a+b,0) / arr.length : 0; }}

function render() {{
  const data = getFiltered();
  const sold = data.filter(r => r.status === 'Sold');
  const active = data.filter(r => r.status === 'Active');

  // ---- KPI cards ----
  const totalListings = data.length;
  const totalSold = sold.length;
  const sellThrough = totalListings ? (100 * totalSold / totalListings) : 0;
  const avgSoldPrice = mean(sold.map(r => r.sold_price));
  const avgDays = mean(sold.map(r => r.days_to_sell));

  document.getElementById('kpiRow').innerHTML = `
    <div class="kpi"><div class="label">Total listings</div><div class="value">${{totalListings.toLocaleString('en-IN')}}</div></div>
    <div class="kpi"><div class="label">Units sold</div><div class="value">${{totalSold.toLocaleString('en-IN')}}</div></div>
    <div class="kpi"><div class="label">Sell-through rate</div><div class="value">${{sellThrough.toFixed(1)}}%</div></div>
    <div class="kpi"><div class="label">Avg sold price</div><div class="value">${{fmtINR(avgSoldPrice)}}</div></div>
    <div class="kpi"><div class="label">Avg days to sell</div><div class="value">${{avgDays.toFixed(1)}}</div></div>
  `;

  // ---- Trend chart (monthly units + avg price) ----
  const months = [...new Set(sold.map(r => r.listing_month))].sort();
  const unitsPerMonth = months.map(m => sold.filter(r => r.listing_month === m).length);
  const avgPricePerMonth = months.map(m => mean(sold.filter(r => r.listing_month === m).map(r => r.sold_price)));

  Plotly.newPlot('chartTrend', [
    {{ x: months, y: unitsPerMonth, type: 'bar', name: 'Units sold', marker: {{ color: '#4c7fee' }}, yaxis: 'y' }},
    {{ x: months, y: avgPricePerMonth, type: 'scatter', mode: 'lines+markers', name: 'Avg price', line: {{ color: '#22c58b' }}, yaxis: 'y2' }},
  ], {{
    ...PLOT_LAYOUT_BASE,
    yaxis: {{ ...PLOT_LAYOUT_BASE.yaxis, title: 'Units' }},
    yaxis2: {{ overlaying: 'y', side: 'right', showgrid: false, title: 'Avg price' }},
  }}, {{ displayModeBar: false, responsive: true }});

  // ---- Avg price by brand ----
  const brandGroups = {{}};
  sold.forEach(r => {{ (brandGroups[r.brand] = brandGroups[r.brand] || []).push(r.sold_price); }});
  const brandNames = Object.keys(brandGroups).sort((a,bb) => mean(brandGroups[bb]) - mean(brandGroups[a]));
  Plotly.newPlot('chartBrand', [{{
    x: brandNames.map(b => mean(brandGroups[b])),
    y: brandNames, type: 'bar', orientation: 'h',
    marker: {{ color: '#7f77dd' }},
  }}], {{ ...PLOT_LAYOUT_BASE, xaxis: {{ ...PLOT_LAYOUT_BASE.xaxis, title: 'Avg sold price' }} }},
  {{ displayModeBar: false, responsive: true }});

  // ---- Sell-through + avg days by city ----
  const cityNames = [...new Set(data.map(r => r.city))].sort();
  const cityListings = cityNames.map(c => data.filter(r => r.city === c).length);
  const citySold = cityNames.map(c => data.filter(r => r.city === c && r.status === 'Sold').length);
  const citySellThrough = cityNames.map((c,i) => cityListings[i] ? 100*citySold[i]/cityListings[i] : 0);
  const cityAvgDays = cityNames.map(c => mean(data.filter(r => r.city === c && r.status === 'Sold').map(r => r.days_to_sell)));

  Plotly.newPlot('chartCity', [
    {{ x: cityNames, y: citySellThrough, type: 'bar', name: 'Sell-through %', marker: {{ color: '#22c58b' }} }},
    {{ x: cityNames, y: cityAvgDays, type: 'scatter', mode: 'lines+markers', name: 'Avg days to sell', line: {{ color: '#f0a63a' }}, yaxis: 'y2' }},
  ], {{
    ...PLOT_LAYOUT_BASE,
    xaxis: {{ ...PLOT_LAYOUT_BASE.xaxis, tickangle: -30 }},
    yaxis: {{ ...PLOT_LAYOUT_BASE.yaxis, title: '%' }},
    yaxis2: {{ overlaying: 'y', side: 'right', showgrid: false, title: 'Days' }},
  }}, {{ displayModeBar: false, responsive: true }});

  // ---- Price by fuel/transmission (box) ----
  const fuels = [...new Set(data.map(r => r.fuel_type))].sort();
  const traces = ['Manual', 'Automatic'].map((t, i) => ({{
    y: fuels.flatMap(f => sold.filter(r => r.fuel_type === f && r.transmission === t).map(r => r.sold_price)),
    x: fuels.flatMap(f => sold.filter(r => r.fuel_type === f && r.transmission === t).map(() => f)),
    type: 'box', name: t, marker: {{ color: i === 0 ? '#4c7fee' : '#e2504a' }},
  }}));
  Plotly.newPlot('chartFuel', traces, {{ ...PLOT_LAYOUT_BASE, boxmode: 'group' }}, {{ displayModeBar: false, responsive: true }});

  // ---- Inventory ageing ----
  const now = new Date('2025-06-30');
  const buckets = {{ '0-15 days': 0, '16-30 days': 0, '31-60 days': 0, '60+ days': 0 }};
  active.forEach(r => {{
    // approximate age from listing_month since day not embedded; use month diff as proxy
    const listed = new Date(r.listing_month + '-15');
    const daysDiff = Math.round((now - listed) / 86400000);
    if (daysDiff <= 15) buckets['0-15 days']++;
    else if (daysDiff <= 30) buckets['16-30 days']++;
    else if (daysDiff <= 60) buckets['31-60 days']++;
    else buckets['60+ days']++;
  }});
  Plotly.newPlot('chartAge', [{{
    x: Object.keys(buckets), y: Object.values(buckets), type: 'bar',
    marker: {{ color: ['#22c58b','#4c7fee','#f0a63a','#e2504a'] }},
  }}], {{ ...PLOT_LAYOUT_BASE, yaxis: {{ ...PLOT_LAYOUT_BASE.yaxis, title: 'Active listings' }} }},
  {{ displayModeBar: false, responsive: true }});
}}

cityFilter.addEventListener('change', render);
brandFilter.addEventListener('change', render);
render();
</script>

</body>
</html>
"""

with open("dashboard/dashboard.html", "w") as f:
    f.write(html)

print(f"Dashboard built -> dashboard/dashboard.html ({len(html)/1024:.0f} KB)")
