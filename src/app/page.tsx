'use client'

import React, { useMemo, useState, useEffect, useRef } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Activity,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  CloudSun,
  Download,
  FileSpreadsheet,
  Gauge,
  LayoutDashboard,
  Menu,
  RotateCcw,
  Settings2,
  Sparkles,
  SunMedium,
  UploadCloud,
  Wind,
  X,
  Zap,
} from 'lucide-react'
import {
  predictYieldLocal,
  predictYieldWithBackend,
  parseWeatherCsv,
  generateSampleCsvContent,
  triggerFileDownload,
  presets,
  type PredictionInputs,
  type Prediction,
  type ImportedDataset,
} from '@/services/predictionService'

const initialInputs: PredictionInputs = {
  ghi: 450,
  windSpeed: 7.5,
  temperature: 18,
  windDirection: 180,
  humidity: 58,
  lagPower: 5,
}

const tooltipStyle = {
  background: '#ffffff',
  border: '1px solid #dce4e8',
  borderRadius: 8,
  color: '#243746',
  fontSize: 12,
  boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
}

type Tab = 'Overview' | 'Forecast' | 'Analytics' | 'Plant Health' | 'Data Import' | 'About System'

function Sidebar({
  active,
  setActive,
  open,
  onClose,
  hasImportedData,
}: {
  active: Tab
  setActive: (tab: Tab) => void
  open: boolean
  onClose: () => void
  hasImportedData: boolean
}) {
  const items: [Tab, React.ElementType][] = [
    ['Overview', LayoutDashboard],
    ['Forecast', Activity],
    ['Analytics', BarChart3],
    ['Plant Health', Gauge],
    ['Data Import', UploadCloud],
  ]

  return (
    <aside
      className={`${
        open ? 'translate-x-0' : '-translate-x-full'
      } fixed inset-y-0 left-0 z-30 flex w-64 flex-col border-r border-slate-200 bg-[#243b53] text-white transition-transform lg:translate-x-0`}
    >
      <div className="flex h-20 items-center justify-between border-b border-white/10 px-6">
        <div className="flex items-center gap-2">
          <span className="grid h-9 w-9 place-items-center rounded-lg bg-[#1d9a78] text-white">
            <Zap className="h-5 w-5" />
          </span>
          <div>
            <div className="text-lg font-bold tracking-tight">
              RENEW<span className="text-[#79c7ae]">GRID</span>
            </div>
            <div className="text-[9px] uppercase tracking-[.18em] text-slate-300">
              Hybrid Energy Yield
            </div>
          </div>
        </div>
        <button className="lg:hidden" onClick={onClose} aria-label="Close navigation">
          <X className="h-5 w-5" />
        </button>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-6">
        {items.map(([label, Icon]) => (
          <button
            key={label}
            onClick={() => {
              setActive(label)
              onClose()
            }}
            className={`flex w-full items-center justify-between rounded-lg px-4 py-3 text-left text-sm transition ${
              active === label
                ? 'bg-white/12 font-semibold text-white'
                : 'text-slate-300 hover:bg-white/8 hover:text-white'
            }`}
          >
            <div className="flex items-center gap-3">
              <Icon className="h-4 w-4" />
              {label}
            </div>
            {label === 'Data Import' && hasImportedData && (
              <span className="h-2 w-2 rounded-full bg-[#1d9a78]" />
            )}
          </button>
        ))}

        <div className="my-6 border-t border-white/10" />

        <button
          onClick={() => {
            setActive('About System')
            onClose()
          }}
          className={`flex w-full items-center gap-3 rounded-lg px-4 py-3 text-left text-sm ${
            active === 'About System'
              ? 'bg-white/12 font-semibold text-white'
              : 'text-slate-300 hover:bg-white/8 hover:text-white'
          }`}
        >
          <Settings2 className="h-4 w-4" />
          About System
        </button>
      </nav>

      <div className="border-t border-white/10 p-5">
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold">
          <span className="pulse-dot h-2 w-2 rounded-full bg-[#63c79f]" />
          Backend Model Online
        </div>
        <p className="text-[11px] leading-5 text-slate-400">
          DTU SOLETE 18 kW Facility · Voting Ensemble
        </p>
      </div>
    </aside>
  )
}

function Panel({
  title,
  description,
  children,
  className = '',
}: {
  title: string
  description?: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <section className={`rounded-lg border border-slate-200 bg-white p-4 shadow-sm sm:p-6 ${className}`}>
      <div className="mb-5">
        <h3 className="font-semibold text-[#243746]">{title}</h3>
        {description && <p className="mt-1 text-xs text-slate-500">{description}</p>}
      </div>
      {children}
    </section>
  )
}

function KPI({
  label,
  value,
  unit,
  icon: Icon,
  tone,
  subtitle = 'Live model calculation',
}: {
  label: string
  value: string
  unit: string
  icon: React.ElementType
  tone: string
  subtitle?: string
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-medium text-slate-500">{label}</span>
        <span className={`grid h-8 w-8 place-items-center rounded-md ${tone}`}>
          <Icon className="h-4 w-4" />
        </span>
      </div>
      <div className="text-2xl font-bold tracking-tight text-[#243746]">
        {value}
        <span className="ml-1 text-sm font-medium text-slate-400">{unit}</span>
      </div>
      <div className="mt-2 text-[11px] text-[#1d9a78]">{subtitle}</div>
    </div>
  )
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  unit: string
  onChange: (value: number) => void
}) {
  return (
    <label className="block">
      <div className="mb-2 flex justify-between text-xs font-medium text-slate-600">
        <span>{label}</span>
        <span className="font-semibold text-[#243b53]">
          {value}
          {unit}
        </span>
      </div>
      <input
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full accent-[#1d9a78]"
      />
    </label>
  )
}

// Overview View
function Overview({
  inputs,
  importedData,
  onClearImported,
  onGoToImport,
}: {
  inputs: PredictionInputs
  importedData: ImportedDataset | null
  onClearImported: () => void
  onGoToImport: () => void
}) {
  const [prediction, setPrediction] = useState<Prediction>(() => predictYieldLocal(inputs))

  useEffect(() => {
    let isMounted = true
    predictYieldWithBackend(inputs).then((res) => {
      if (isMounted) setPrediction(res)
    })
    return () => {
      isMounted = false
    }
  }, [inputs])

  const chartData = useMemo(() => {
    if (importedData && importedData.rows.length > 0) {
      return importedData.rows.slice(0, 24).map((r) => ({
        label: r.timestamp.includes(' ') ? r.timestamp.split(' ')[1].slice(0, 5) : r.timestamp,
        solar: r.predictedSolar,
        wind: r.predictedWind,
        total: r.predictedTotal,
      }))
    }
    return prediction.forecast.map((p, i) => ({
      label: `${String(i * 3).padStart(2, '0')}:00`,
      solar: p.solar,
      wind: p.wind,
      total: p.total,
    }))
  }, [importedData, prediction])

  const currentTotal = importedData ? importedData.avgTotalKW : prediction.total
  const currentSolar = importedData ? importedData.avgSolarKW : prediction.solar
  const currentWind = importedData ? importedData.avgWindKW : prediction.wind
  const todayKWh = importedData ? importedData.totalEnergyKWh : Number((prediction.total * 8.6).toFixed(0))

  const mix = [
    { name: 'Solar', value: Math.max(0.1, currentSolar), color: '#e3b341' },
    { name: 'Wind', value: Math.max(0.1, currentWind), color: '#2f8f72' },
  ]

  const handleExportOverview = () => {
    const rows = [
      'timestamp,solar_kw,wind_kw,hybrid_total_kw',
      ...chartData.map((d) => `${d.label},${d.solar},${d.wind},${d.total}`),
    ].join('\n')
    triggerFileDownload('solete_generation_overview.csv', rows)
  }

  return (
    <>
      <div className="mb-6 flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
        <div>
          <div className="mb-1 text-xs font-semibold uppercase tracking-[.16em] text-[#1d9a78]">
            Portfolio / SOLETE Facility
          </div>
          <h2 className="text-2xl font-bold tracking-tight">Generation Performance</h2>
          {importedData ? (
            <div className="mt-2 flex items-center gap-2">
              <span className="rounded-md bg-[#e8f5ef] px-2.5 py-1 text-xs font-semibold text-[#1d9a78]">
                Displaying uploaded dataset: {importedData.fileName} ({importedData.rows.length} rows)
              </span>
              <button
                onClick={onClearImported}
                className="flex items-center gap-1 text-xs text-slate-500 hover:text-red-600"
              >
                <RotateCcw className="h-3 w-3" /> Reset to live inputs
              </button>
            </div>
          ) : (
            <p className="mt-1 text-xs text-slate-500">
              Co-located 7 kW PV Array and Gaia 11 kW Wind Turbine output.
            </p>
          )}
        </div>
        <div className="flex gap-2">
          {!importedData && (
            <button
              onClick={onGoToImport}
              className="flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm hover:bg-slate-50"
            >
              <UploadCloud className="h-3.5 w-3.5 text-[#1d9a78]" />
              Import CSV
            </button>
          )}
          <button
            onClick={handleExportOverview}
            className="flex items-center gap-1.5 rounded-md bg-[#243b53] px-3 py-2 text-xs font-semibold text-white shadow-sm hover:bg-[#1a2c3f]"
          >
            <Download className="h-3.5 w-3.5" />
            Export Overview CSV
          </button>
        </div>
      </div>

      <section className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPI
          label={importedData ? 'Average Hybrid Power' : 'Current Hybrid Power'}
          value={currentTotal.toFixed(1)}
          unit="kW"
          icon={Zap}
          tone="bg-[#e8f5ef] text-[#1d9a78]"
          subtitle={importedData ? 'Computed across dataset' : 'Live ML calculation'}
        />
        <KPI
          label={importedData ? 'Average Solar Power' : 'Current Solar Power'}
          value={currentSolar.toFixed(1)}
          unit="kW"
          icon={CloudSun}
          tone="bg-[#fff5d9] text-[#b88914]"
        />
        <KPI
          label={importedData ? 'Average Wind Power' : 'Current Wind Power'}
          value={currentWind.toFixed(1)}
          unit="kW"
          icon={Wind}
          tone="bg-[#e7f0f6] text-[#39708b]"
        />
        <KPI
          label={importedData ? 'Dataset Total Energy' : "Today's Energy Yield"}
          value={todayKWh.toString()}
          unit="kWh"
          icon={Gauge}
          tone="bg-[#eef0fb] text-[#5967a5]"
        />
      </section>

      <Panel
        title="Hybrid Generation Profile"
        description={
          importedData
            ? `Timeline generated from ${importedData.fileName}`
            : 'Solar and wind contribution across the 24-hour cycle'
        }
      >
        <ResponsiveContainer width="100%" height={330}>
          <LineChart data={chartData}>
            <CartesianGrid stroke="#edf1f2" vertical={false} />
            <XAxis dataKey="label" tick={{ fill: '#84929b', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#84929b', fontSize: 11 }} axisLine={false} tickLine={false} unit=" kW" />
            <Tooltip contentStyle={tooltipStyle} />
            <Line dataKey="total" stroke="#243b53" strokeWidth={3} dot={false} name="Hybrid Total" />
            <Line dataKey="solar" stroke="#e3b341" strokeWidth={2} dot={false} name="Solar Output" />
            <Line dataKey="wind" stroke="#2f8f72" strokeWidth={2} dot={false} name="Wind Output" />
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      <div className="mt-6 grid gap-6 xl:grid-cols-[1.2fr_.8fr]">
        <Panel title="Energy Yield by Day" description="Simulated 7-day plant output · MWh">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart
              data={[
                { day: 'Mon', solar: 19.2, wind: 12.4 },
                { day: 'Tue', solar: 22.1, wind: 14.8 },
                { day: 'Wed', solar: 17.5, wind: 19.2 },
                { day: 'Thu', solar: 24.3, wind: 9.6 },
                { day: 'Fri', solar: 21.0, wind: 15.3 },
                { day: 'Sat', solar: 18.4, wind: 18.1 },
                { day: 'Sun', solar: 23.8, wind: 13.5 },
              ]}
            >
              <CartesianGrid stroke="#edf1f2" vertical={false} />
              <XAxis dataKey="day" tick={{ fill: '#84929b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#84929b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="solar" stackId="energy" fill="#e3b341" name="Solar" />
              <Bar dataKey="wind" stackId="energy" fill="#2f8f72" name="Wind" />
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Current Energy Mix" description="Source share of total hybrid generation">
          <div className="relative">
            <ResponsiveContainer width="100%" height={210}>
              <PieChart>
                <Pie data={mix} dataKey="value" innerRadius={58} outerRadius={80} paddingAngle={3}>
                  {mix.map((item) => (
                    <Cell key={item.name} fill={item.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
              <div>
                <div className="text-2xl font-bold">{currentTotal.toFixed(1)}</div>
                <div className="text-[10px] text-slate-400">kW Total</div>
              </div>
            </div>
          </div>
          <div className="space-y-2 text-xs">
            {mix.map((item) => (
              <div key={item.name} className="flex justify-between">
                <span>
                  <i className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: item.color }} />
                  {item.name}
                </span>
                <span className="font-semibold">
                  {currentTotal > 0 ? Math.round((item.value / currentTotal) * 100) : 50}%
                </span>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </>
  )
}

// Forecast & Simulation View
function ForecastView({
  inputs,
  setInputs,
}: {
  inputs: PredictionInputs
  setInputs: (next: PredictionInputs) => void
}) {
  const [prediction, setPrediction] = useState<Prediction>(() => predictYieldLocal(inputs))
  const [isUpdating, setIsUpdating] = useState(false)

  useEffect(() => {
    let isMounted = true
    setIsUpdating(true)
    predictYieldWithBackend(inputs).then((res) => {
      if (isMounted) {
        setPrediction(res)
        setIsUpdating(false)
      }
    })
    return () => {
      isMounted = false
    }
  }, [inputs])

  const forecast = prediction.forecast.map((point, index) => ({
    ...point,
    label: `${String(index * 3).padStart(2, '0')}:00`,
  }))

  const handleExportForecast = () => {
    const csvContent = [
      'hour,solar_kw,wind_kw,total_kw,ghi_w_m2,wind_speed_m_s',
      ...prediction.forecast.map(
        (p) => `${p.hour},${p.solar},${p.wind},${p.total},${p.ghi},${p.windSpeed}`
      ),
    ].join('\n')
    triggerFileDownload('hybrid_yield_forecast_24h.csv', csvContent)
  }

  return (
    <>
      <div className="mb-6 flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
        <div>
          <div className="mb-1 text-xs font-semibold uppercase tracking-[.16em] text-[#1d9a78]">
            Interactive Predictor
          </div>
          <h2 className="text-2xl font-bold tracking-tight">Weather & Yield Forecast</h2>
          <p className="mt-1 text-sm text-slate-500">
            Adjust meteorological variables to evaluate plant performance in real time.
          </p>
        </div>
        <button
          onClick={handleExportForecast}
          className="flex items-center gap-1.5 rounded-md bg-[#243b53] px-3 py-2 text-xs font-semibold text-white shadow-sm hover:bg-[#1a2c3f]"
        >
          <Download className="h-3.5 w-3.5" />
          Export Forecast CSV
        </button>
      </div>

      <div className="grid gap-6 xl:grid-cols-[.8fr_1.2fr]">
        <Panel title="Weather Conditions" description="Exogenous inputs to the hybrid ensemble model">
          <div className="space-y-6">
            <Slider
              label="Solar Irradiance (GHI)"
              value={inputs.ghi}
              min={0}
              max={1000}
              step={10}
              unit=" W/m²"
              onChange={(value) => setInputs({ ...inputs, ghi: value })}
            />
            <Slider
              label="Wind Speed"
              value={inputs.windSpeed}
              min={0}
              max={25}
              step={0.5}
              unit=" m/s"
              onChange={(value) => setInputs({ ...inputs, windSpeed: value })}
            />
            <Slider
              label="Ambient Temperature"
              value={inputs.temperature}
              min={-5}
              max={40}
              step={1}
              unit="°C"
              onChange={(value) => setInputs({ ...inputs, temperature: value })}
            />
            <Slider
              label="Past Hour Generation"
              value={inputs.lagPower}
              min={0}
              max={18}
              step={0.1}
              unit=" kW"
              onChange={(value) => setInputs({ ...inputs, lagPower: value })}
            />
            <div>
              <div className="mb-2 flex justify-between text-xs font-medium text-slate-600">
                <span>Wind Direction</span>
                <span className="font-semibold text-[#243b53]">{inputs.windDirection}°</span>
              </div>
              <div className="flex items-center gap-5">
                <div className="relative grid h-20 w-20 shrink-0 place-items-center rounded-full border-2 border-[#dce4e8] text-[10px] font-semibold text-slate-400">
                  <span>N</span>
                  <span className="absolute bottom-1">S</span>
                  <span className="absolute left-1">W</span>
                  <span className="absolute right-1">E</span>
                  <span
                    className="absolute h-7 w-0.5 origin-bottom bg-[#1d9a78] transition-transform duration-200"
                    style={{ transform: `rotate(${inputs.windDirection}deg) translateY(-5px)` }}
                  />
                </div>
                <input
                  aria-label="Wind Direction"
                  type="range"
                  min="0"
                  max="360"
                  step="5"
                  value={inputs.windDirection}
                  onChange={(event) => setInputs({ ...inputs, windDirection: Number(event.target.value) })}
                  className="h-1.5 w-full accent-[#1d9a78]"
                />
              </div>
            </div>
          </div>
        </Panel>

        <Panel
          title="Predicted Hybrid Output"
          description={`Engineered with DTU SOLETE Ensemble · ${prediction.source}`}
        >
          <div className="grid items-center gap-6 sm:grid-cols-[180px_1fr]">
            <div
              className="relative mx-auto grid h-44 w-44 place-items-center rounded-full transition-all duration-300"
              style={{
                background: `conic-gradient(#1d9a78 ${Math.min(prediction.total / 18, 1) * 360}deg, #e7eff0 0)`,
              }}
            >
              <div className="grid h-36 w-36 place-items-center rounded-full bg-white text-center shadow-inner">
                <div>
                  <div className="text-3xl font-bold text-[#243b53]">
                    {isUpdating ? '...' : prediction.total.toFixed(2)}
                  </div>
                  <div className="text-xs text-slate-500">kW Output</div>
                  <div className="mt-1 text-[10px] text-[#1d9a78] font-semibold">
                    {prediction.capacityFactor.toFixed(1)}% Cap. Factor
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="rounded-lg bg-[#fff8df] p-4">
                <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                  <SunMedium className="h-4 w-4 text-[#b88914]" />
                  Solar PV Output (7 kW Array)
                </div>
                <div className="mt-1 text-2xl font-bold text-[#243746]">
                  {prediction.solar.toFixed(2)} <span className="text-sm font-medium text-slate-400">kW</span>
                </div>
              </div>
              <div className="rounded-lg bg-[#e8f5ef] p-4">
                <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                  <Wind className="h-4 w-4 text-[#2f8f72]" />
                  Wind Turbine Output (Gaia 11 kW)
                </div>
                <div className="mt-1 text-2xl font-bold text-[#243746]">
                  {prediction.wind.toFixed(2)} <span className="text-sm font-medium text-slate-400">kW</span>
                </div>
              </div>
            </div>
          </div>
        </Panel>
      </div>

      <div className="my-6">
        <div className="mb-2 text-xs font-semibold text-slate-600">Quick Test Scenarios:</div>
        <div className="flex flex-wrap gap-2">
          {Object.keys(presets).map((name) => (
            <button
              key={name}
              onClick={() => setInputs(presets[name])}
              className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600 shadow-xs transition hover:border-[#1d9a78] hover:text-[#1d9a78]"
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      <Panel
        title="24-Hour Day-Ahead Generation Forecast"
        description="Composite diurnal projection computed across varying solar altitude and wind cycles"
      >
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={forecast}>
            <CartesianGrid stroke="#edf1f2" vertical={false} />
            <XAxis dataKey="label" tick={{ fill: '#84929b', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#84929b', fontSize: 11 }} axisLine={false} tickLine={false} unit=" kW" />
            <Tooltip contentStyle={tooltipStyle} />
            <Area type="monotone" dataKey="total" stroke="#243b53" fill="#dce9e7" strokeWidth={3} name="Hybrid Output" />
          </AreaChart>
        </ResponsiveContainer>
      </Panel>
    </>
  )
}

// Analytics View
function AnalyticsView() {
  const complementarity = Array.from({ length: 12 }, (_, i) => ({
    hour: `${String(i * 2).padStart(2, '0')}:00`,
    solar: Math.max(0, Math.sin(((i - 2) / 6) * Math.PI) * 6.8),
    wind: 4.2 + Math.max(0, Math.sin(((i + 2) / 6) * Math.PI)) * 5.4,
  }))

  const weekly = [
    { day: 'Mon', total: 28.5 },
    { day: 'Tue', total: 32.1 },
    { day: 'Wed', total: 29.8 },
    { day: 'Thu', total: 35.4 },
    { day: 'Fri', total: 31.2 },
    { day: 'Sat', total: 36.7 },
    { day: 'Sun', total: 30.3 },
  ]

  return (
    <>
      <div className="mb-6">
        <div className="mb-1 text-xs font-semibold uppercase tracking-[.16em] text-[#1d9a78]">
          Production Insights
        </div>
        <h2 className="text-2xl font-bold tracking-tight">Energy Production Analytics</h2>
        <p className="mt-1 text-sm text-slate-500">
          Cross-source complementary dynamics and operating windows.
        </p>
      </div>

      <div className="mb-6 grid gap-4 sm:grid-cols-3">
        <KPI label="Weekly Energy Total" value="224.8" unit="MWh" icon={Zap} tone="bg-[#e8f5ef] text-[#1d9a78]" />
        <KPI label="Average Capacity Factor" value="61.4" unit="%" icon={Gauge} tone="bg-[#fff5d9] text-[#b88914]" />
        <KPI label="Renewable Share" value="100" unit="%" icon={SunMedium} tone="bg-[#e7f0f6] text-[#39708b]" />
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.3fr_.7fr]">
        <Panel
          title="Solar & Wind Complementarity"
          description="Wind ramps up as solar irradiance declines in the evening hours"
        >
          <ResponsiveContainer width="100%" height={310}>
            <ComposedChart data={complementarity}>
              <CartesianGrid stroke="#edf1f2" vertical={false} />
              <XAxis dataKey="hour" tick={{ fill: '#84929b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#84929b', fontSize: 11 }} axisLine={false} tickLine={false} unit=" kW" />
              <Tooltip contentStyle={tooltipStyle} />
              <Area dataKey="solar" type="monotone" fill="#fff1b8" stroke="#e3b341" name="Solar" />
              <Line dataKey="wind" type="monotone" stroke="#2f8f72" strokeWidth={3} name="Wind" />
            </ComposedChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Peak Generation Windows" description="Physical dispatch characteristics">
          <div className="space-y-3">
            <div className="rounded-lg bg-[#fff8df] p-4">
              <div className="flex items-center gap-2 text-xs font-semibold text-[#8a6812]">
                <SunMedium className="h-4 w-4" />
                Peak Solar Hours
              </div>
              <div className="mt-2 text-lg font-bold">11:00 – 15:00</div>
              <p className="mt-1 text-xs text-slate-500">Highest solar elevation and clear-sky capture</p>
            </div>
            <div className="rounded-lg bg-[#e8f5ef] p-4">
              <div className="flex items-center gap-2 text-xs font-semibold text-[#237b60]">
                <Wind className="h-4 w-4" />
                Peak Wind Hours
              </div>
              <div className="mt-2 text-lg font-bold">18:00 – 04:00</div>
              <p className="mt-1 text-xs text-slate-500">Strong coastal nocturnal wind shear</p>
            </div>
          </div>
        </Panel>
      </div>

      <div className="mt-6">
        <Panel title="Weekly Energy Totals" description="Total delivered energy by day · MWh">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={weekly}>
              <CartesianGrid stroke="#edf1f2" vertical={false} />
              <XAxis dataKey="day" tick={{ fill: '#84929b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#84929b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="total" fill="#2f8f72" radius={[4, 4, 0, 0]} name="MWh Delivered" />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>
    </>
  )
}

// Plant Health View
function HealthView() {
  const cards = [
    ['Solar PV Array & Inverter', 'Active / Optimal', 'Efficiency: 98.4% · Inverter Temp: 38°C · 7 kW Capacity', SunMedium],
    ['Wind Turbine · Gaia 11 kW', 'Operating Normally', 'Pitch: 0° · Rotor: 52 RPM · 11 kW Capacity', Wind],
    ['Point of Common Coupling (PCC)', 'Synchronized to Grid', 'Stable Danish Grid · 50.02 Hz · 18 kW Transformer Limit', Zap],
    ['Met-Mast Weather Station', 'Online & Streaming', 'GHI, POA Irradiance, Wind Vector (u,v), Temp, Humidity', Activity],
  ] as const

  return (
    <>
      <div className="mb-6">
        <div className="mb-1 text-xs font-semibold uppercase tracking-[.16em] text-[#1d9a78]">
          Operations
        </div>
        <h2 className="text-2xl font-bold tracking-tight">Plant Health & Diagnostics</h2>
        <p className="mt-1 text-sm text-slate-500">
          Continuous equipment status monitoring for the DTU SOLETE facility.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {cards.map(([title, status, detail, Icon]) => (
          <div key={title} className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-5 flex items-start justify-between">
              <div className="grid h-11 w-11 place-items-center rounded-lg bg-[#e8f5ef] text-[#1d9a78]">
                <Icon className="h-5 w-5" />
              </div>
              <span className="flex items-center gap-1.5 rounded-full bg-[#e8f5ef] px-2.5 py-1 text-[11px] font-semibold text-[#237b60]">
                <CheckCircle2 className="h-3.5 w-3.5" />
                Operational
              </span>
            </div>
            <h3 className="font-semibold text-[#243746]">{title}</h3>
            <div className="mt-2 text-sm font-semibold text-[#1d9a78]">{status}</div>
            <p className="mt-2 text-xs text-slate-500">{detail}</p>
          </div>
        ))}
      </div>
    </>
  )
}

// Data Import View (Fully Functional CSV Upload, Parsing, Preview & Template Download)
function ImportView({
  importedData,
  setImportedData,
  onApplyData,
}: {
  importedData: ImportedDataset | null
  setImportedData: (data: ImportedDataset | null) => void
  onApplyData: () => void
}) {
  const [dragActive, setDragActive] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const processFile = (file: File) => {
    setErrorMsg(null)
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setErrorMsg('Please upload a valid .csv file.')
      return
    }

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string
        if (!text) {
          setErrorMsg('The file is empty.')
          return
        }
        const parsed = parseWeatherCsv(text, file.name)
        setImportedData(parsed)
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Unknown parsing error'
        setErrorMsg(`Failed to parse CSV: ${message}`)
      }
    }
    reader.onerror = () => {
      setErrorMsg('Error reading file from disk.')
    }
    reader.readAsText(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0])
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(true)
  }

  const handleDragLeave = () => {
    setDragActive(false)
  }

  const handleDownloadSample = () => {
    const csvContent = generateSampleCsvContent()
    triggerFileDownload('solete_sample_weather.csv', csvContent)
  }

  const handleLoadSampleDataset = () => {
    const csvContent = generateSampleCsvContent()
    const parsed = parseWeatherCsv(csvContent, 'sample_solete_24h.csv')
    setImportedData(parsed)
    setErrorMsg(null)
  }

  return (
    <>
      <div className="mb-6">
        <div className="mb-1 text-xs font-semibold uppercase tracking-[.16em] text-[#1d9a78]">
          Connected Data
        </div>
        <h2 className="text-2xl font-bold tracking-tight">Data Import & Batch Yield Forecast</h2>
        <p className="mt-1 text-sm text-slate-500">
          Upload custom weather logs or sensor recordings to evaluate yield across any time horizon.
        </p>
      </div>

      <Panel
        title="Upload Weather Forecast CSV / Sensor Log"
        description="CSV files are automatically validated and processed through the trained hybrid energy model."
      >
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`grid min-h-64 place-items-center rounded-lg border-2 border-dashed p-8 text-center transition ${
            dragActive
              ? 'border-[#1d9a78] bg-[#e8f5ef]'
              : 'border-[#b8d5ca] bg-[#f6fbf8] hover:border-[#1d9a78]'
          }`}
        >
          <UploadCloud className="mb-3 h-12 w-12 text-[#1d9a78]" />
          <h3 className="font-semibold text-lg text-[#243746]">Drop your CSV file here</h3>
          <p className="mt-2 max-w-md text-xs leading-5 text-slate-500">
            Accepts any CSV with weather columns such as <code>irradiance / ghi</code>, <code>wind_speed</code>, and <code>temperature</code>.
          </p>

          <div className="mt-5 flex flex-wrap justify-center gap-3">
            <label className="cursor-pointer rounded-md bg-[#243b53] px-4 py-2.5 text-xs font-semibold text-white shadow-sm hover:bg-[#1a2c3f]">
              Choose CSV File
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    processFile(e.target.files[0])
                  }
                }}
                className="sr-only"
              />
            </label>

            <button
              onClick={handleLoadSampleDataset}
              className="flex items-center gap-1.5 rounded-md border border-[#1d9a78] bg-white px-4 py-2.5 text-xs font-semibold text-[#1d9a78] shadow-sm hover:bg-[#f0f9f5]"
            >
              <Sparkles className="h-4 w-4" />
              Load Sample 24h SOLETE Dataset
            </button>
          </div>
        </div>

        {errorMsg && (
          <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-xs text-red-700">
            {errorMsg}
          </div>
        )}

        <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
          <button
            onClick={handleDownloadSample}
            className="flex items-center gap-2 rounded-md border border-slate-200 px-4 py-2.5 text-xs font-semibold text-slate-600 hover:border-[#1d9a78] hover:text-[#1d9a78]"
          >
            <Download className="h-4 w-4" />
            Download Sample Weather Template (.csv)
          </button>

          {importedData && (
            <button
              onClick={onApplyData}
              className="flex items-center gap-2 rounded-md bg-[#1d9a78] px-4 py-2.5 text-xs font-semibold text-white shadow-sm hover:bg-[#167d61]"
            >
              <FileSpreadsheet className="h-4 w-4" />
              View Processed Dataset in Overview →
            </button>
          )}
        </div>
      </Panel>

      {importedData && (
        <div className="mt-8 space-y-6">
          <Panel
            title={`Batch Processing Summary: ${importedData.fileName}`}
            description={`Model computed yield forecasts for all ${importedData.rows.length} rows`}
          >
            <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <KPI
                label="Total Predicted Energy"
                value={importedData.totalEnergyKWh.toString()}
                unit="kWh"
                icon={Zap}
                tone="bg-[#e8f5ef] text-[#1d9a78]"
                subtitle={`${importedData.rows.length} timestamps evaluated`}
              />
              <KPI
                label="Peak Power Output"
                value={importedData.peakPowerKW.toString()}
                unit="kW"
                icon={Gauge}
                tone="bg-[#fff5d9] text-[#b88914]"
                subtitle="Transformer limit: 18 kW"
              />
              <KPI
                label="Average Solar Share"
                value={importedData.avgSolarKW.toString()}
                unit="kW"
                icon={SunMedium}
                tone="bg-[#fff5d9] text-[#b88914]"
                subtitle="PV Array contribution"
              />
              <KPI
                label="Average Wind Share"
                value={importedData.avgWindKW.toString()}
                unit="kW"
                icon={Wind}
                tone="bg-[#e7f0f6] text-[#39708b]"
                subtitle="Gaia Turbine contribution"
              />
            </div>

            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-left text-xs text-slate-700">
                <thead className="border-b border-slate-200 bg-slate-50 text-[11px] font-bold uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="p-3">Timestamp</th>
                    <th className="p-3">Irradiance (W/m²)</th>
                    <th className="p-3">Wind Speed (m/s)</th>
                    <th className="p-3">Temp (°C)</th>
                    <th className="p-3 text-[#b88914]">Pred. Solar (kW)</th>
                    <th className="p-3 text-[#2f8f72]">Pred. Wind (kW)</th>
                    <th className="p-3 font-bold text-[#243b53]">Pred. Total (kW)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {importedData.rows.slice(0, 10).map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-50">
                      <td className="p-3 font-medium">{row.timestamp}</td>
                      <td className="p-3">{row.ghi}</td>
                      <td className="p-3">{row.windSpeed}</td>
                      <td className="p-3">{row.temperature}</td>
                      <td className="p-3 font-semibold text-[#b88914]">{row.predictedSolar.toFixed(2)}</td>
                      <td className="p-3 font-semibold text-[#2f8f72]">{row.predictedWind.toFixed(2)}</td>
                      <td className="p-3 font-bold text-[#243b53]">{row.predictedTotal.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {importedData.rows.length > 10 && (
                <div className="border-t border-slate-100 bg-slate-50 p-2.5 text-center text-xs text-slate-500">
                  Showing first 10 rows of {importedData.rows.length} total rows.
                </div>
              )}
            </div>
          </Panel>
        </div>
      )}
    </>
  )
}

// About View
function AboutView() {
  const cards = [
    ['Solar Array', '7 kW PV System', 'Multi-angle tracking for enhanced daily capture', SunMedium],
    ['Wind Generator', 'Gaia 11 kW Wind Turbine', 'Installed on a 15m mast with pitch regulation', Wind],
    ['Facility Location', 'DTU Risø Research Station', 'Roskilde, Denmark · 457 continuous days of telemetry', Settings2],
  ] as const

  return (
    <>
      <div className="mb-6">
        <div className="mb-1 text-xs font-semibold uppercase tracking-[.16em] text-[#1d9a78]">
          The Facility
        </div>
        <h2 className="text-2xl font-bold tracking-tight">About the SOLETE System</h2>
        <p className="mt-1 text-sm text-slate-500">
          Physical plant specifications behind the RenewGrid hybrid forecasting workspace.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {cards.map(([title, value, detail, Icon]) => (
          <div key={title} className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-6 grid h-12 w-12 place-items-center rounded-lg bg-[#e8f5ef] text-[#1d9a78]">
              <Icon className="h-6 w-6" />
            </div>
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</div>
            <h3 className="mt-2 text-lg font-bold text-[#243746]">{value}</h3>
            <p className="mt-2 text-sm leading-6 text-slate-500">{detail}</p>
          </div>
        ))}
      </div>

      <div className="mt-6 rounded-lg border border-[#b8d5ca] bg-[#f6fbf8] p-6">
        <div className="flex items-start gap-3">
          <CircleHelp className="mt-0.5 h-5 w-5 shrink-0 text-[#1d9a78]" />
          <div>
            <h3 className="font-semibold text-[#243746]">Co-Located Physical Connection</h3>
            <p className="mt-1 text-sm leading-6 text-slate-600">
              Unlike regional grid aggregations, the DTU Risø SOLETE installation places both wind turbine and solar panels
              on the same point of common coupling. This allows machine learning models to capture true micro-climate weather
              effects and cross-generation smoothing.
            </p>
          </div>
        </div>
      </div>
    </>
  )
}

// Main Page Component
export default function Home() {
  const [active, setActive] = useState<Tab>('Overview')
  const [mobileNav, setMobileNav] = useState(false)
  const [inputs, setInputs] = useState<PredictionInputs>(initialInputs)
  const [importedData, setImportedData] = useState<ImportedDataset | null>(null)

  const content =
    active === 'Overview' ? (
      <Overview
        inputs={inputs}
        importedData={importedData}
        onClearImported={() => setImportedData(null)}
        onGoToImport={() => setActive('Data Import')}
      />
    ) : active === 'Forecast' ? (
      <ForecastView inputs={inputs} setInputs={setInputs} />
    ) : active === 'Analytics' ? (
      <AnalyticsView />
    ) : active === 'Plant Health' ? (
      <HealthView />
    ) : active === 'Data Import' ? (
      <ImportView
        importedData={importedData}
        setImportedData={setImportedData}
        onApplyData={() => setActive('Overview')}
      />
    ) : (
      <AboutView />
    )

  return (
    <div className="min-h-screen bg-[#f4f8f8] text-[#243746]">
      <Sidebar
        active={active}
        setActive={setActive}
        open={mobileNav}
        onClose={() => setMobileNav(false)}
        hasImportedData={importedData !== null}
      />

      <div className="lg:pl-64">
        <header className="flex h-20 items-center justify-between border-b border-slate-200 bg-white px-5 sm:px-8">
          <div className="flex items-center gap-3">
            <button
              className="lg:hidden"
              onClick={() => setMobileNav(true)}
              aria-label="Open navigation"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div>
              <h1 className="text-lg font-bold tracking-tight sm:text-xl">
                {active === 'Overview' ? 'Renewable Energy Overview' : active}
              </h1>
              <p className="hidden text-xs text-slate-500 sm:block">
                Hybrid generation monitoring and yield forecasting
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden items-center gap-2 rounded-full bg-[#e8f5ef] px-3 py-1.5 text-xs font-semibold text-[#237b60] sm:flex">
              <span className="h-2 w-2 rounded-full bg-[#2f8f72] animate-pulse" />
              SOLETE 18 kW Model Online
            </span>
          </div>
        </header>

        <main className="mx-auto max-w-[1500px] p-5 sm:p-8">
          {content}
          <footer className="mt-8 flex flex-col justify-between gap-3 border-t border-slate-200 py-6 text-xs text-slate-500 sm:flex-row">
            <span>RenewGrid Monitoring Platform · SOLETE Co-Located Facility</span>
            <span className="flex items-center gap-2">
              <ChevronRight className="h-3.5 w-3.5" />
              {active} Workspace
            </span>
          </footer>
        </main>
      </div>
    </div>
  )
}
