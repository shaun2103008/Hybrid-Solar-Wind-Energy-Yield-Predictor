export const HF_SPACE_ENDPOINT = 'https://uwouldnever-hybrid-energy-predictor.hf.space'

export type PredictionInputs = {
  ghi: number
  windSpeed: number
  temperature: number
  windDirection: number
  humidity: number
  lagPower: number
}

export type Prediction = {
  solar: number
  wind: number
  total: number
  ratio: number
  capacityFactor: number
  source: 'Hugging Face ML Model' | 'Calibrated Physical Engine'
  forecast: ForecastPoint[]
}

export type ForecastPoint = {
  hour: string
  solar: number
  wind: number
  total: number
  ghi: number
  windSpeed: number
}

export type ImportedRow = {
  timestamp: string
  ghi: number
  windSpeed: number
  temperature: number
  windDirection: number
  predictedSolar: number
  predictedWind: number
  predictedTotal: number
}

export type ImportedDataset = {
  fileName: string
  rows: ImportedRow[]
  totalEnergyKWh: number
  peakPowerKW: number
  avgSolarKW: number
  avgWindKW: number
  avgTotalKW: number
  avgCapacityFactor: number
}

export const presets: Record<string, PredictionInputs> = {
  'Sunny Calm Day': { ghi: 850, windSpeed: 4.0, temperature: 22, windDirection: 135, humidity: 45, lagPower: 6.2 },
  'Stormy High Wind': { ghi: 180, windSpeed: 19.0, temperature: 12, windDirection: 270, humidity: 85, lagPower: 8.5 },
  'Overcast Breeze': { ghi: 260, windSpeed: 10.5, temperature: 16, windDirection: 210, humidity: 70, lagPower: 4.4 },
  'Night-time Wind Surge': { ghi: 0, windSpeed: 16.0, temperature: 8, windDirection: 45, humidity: 80, lagPower: 5.8 },
}

export function predictYieldLocal(input: PredictionInputs): Prediction {
  // Calibrated to DTU SOLETE facility: 7 kW Solar PV + 11 kW Gaia Wind Turbine
  const solar = Math.min(7.0, Math.max(0, (input.ghi / 1000) * 6.85 * (1 - Math.max(0, input.temperature - 25) * 0.004)))
  
  // Wind cubic power law with cut-in (3 m/s), rated (12 m/s), and cut-out (25 m/s)
  let wind = 0
  if (input.windSpeed >= 3.0 && input.windSpeed <= 25.0) {
    if (input.windSpeed < 12.0) {
      wind = Math.pow(input.windSpeed / 12.0, 3.0) * 11.0
    } else {
      wind = 11.0
    }
  }
  
  // Total hybrid power output (accounting for transformer/inverter grid connection limits ~18 kW)
  const total = Math.min(18.0, Number((solar + wind).toFixed(2)))
  const ratio = total > 0 ? Number(((solar / total) * 100).toFixed(1)) : 0
  const capacityFactor = Number(((total / 18.0) * 100).toFixed(1))

  const forecast: ForecastPoint[] = Array.from({ length: 24 }, (_, hour) => {
    // Solar diurnal sun elevation curve
    const daylight = Math.max(0, Math.sin(((hour - 6) / 13) * Math.PI))
    const solarPoint = Math.min(7.0, Math.max(0, solar * daylight * (0.95 + Math.sin(hour * 1.5) * 0.05)))
    
    // Wind diurnal profile (higher in evening/night)
    const windVariation = 1.0 + Math.sin(((hour - 14) / 24) * 2 * Math.PI) * 0.2
    const windSpeedAtHour = Math.max(0, input.windSpeed * windVariation)
    let windPoint = 0
    if (windSpeedAtHour >= 3.0 && windSpeedAtHour <= 25.0) {
      windPoint = windSpeedAtHour < 12.0 ? Math.pow(windSpeedAtHour / 12.0, 3.0) * 11.0 : 11.0
    }

    const totalPoint = Math.min(18.0, solarPoint + windPoint)
    return {
      hour: `${String(hour).padStart(2, '0')}:00`,
      solar: Number(solarPoint.toFixed(2)),
      wind: Number(windPoint.toFixed(2)),
      total: Number(totalPoint.toFixed(2)),
      ghi: Math.round(input.ghi * daylight),
      windSpeed: Number(windSpeedAtHour.toFixed(1)),
    }
  })

  return {
    solar: Number(solar.toFixed(2)),
    wind: Number(wind.toFixed(2)),
    total,
    ratio,
    capacityFactor,
    source: 'Calibrated Physical Engine',
    forecast,
  }
}

export async function predictYieldWithBackend(input: PredictionInputs): Promise<Prediction> {
  const localPrediction = predictYieldLocal(input)
  try {
    const currentHour = new Date().getHours()
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 3500)

    const response = await fetch(`${HF_SPACE_ENDPOINT}/run/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data: [input.windSpeed, input.ghi, input.temperature, currentHour],
      }),
      signal: controller.signal,
    })

    clearTimeout(timeoutId)

    if (response.ok) {
      const result = await response.json()
      if (result && Array.isArray(result.data) && result.data.length >= 3) {
        const total = Math.max(0, Number(result.data[0]))
        const solar = Math.max(0, Number(result.data[1]))
        const wind = Math.max(0, Number(result.data[2]))
        const ratio = total > 0 ? Number(((solar / total) * 100).toFixed(1)) : 0
        const capacityFactor = Number(((total / 18.0) * 100).toFixed(1))

        return {
          ...localPrediction,
          solar: Number(solar.toFixed(2)),
          wind: Number(wind.toFixed(2)),
          total: Number(total.toFixed(2)),
          ratio,
          capacityFactor,
          source: 'Hugging Face ML Model',
        }
      }
    }
  } catch {
    // Graceful fallback to local physical SOLETE engine
  }

  return localPrediction
}

// Flexible CSV Parser for any Weather / Sensor file
export function parseWeatherCsv(csvText: string, fileName: string): ImportedDataset {
  const lines = csvText.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0)
  if (lines.length < 2) {
    throw new Error('The CSV file does not contain enough rows.')
  }

  // Detect delimiter (comma or semicolon)
  const firstLine = lines[0]
  const delimiter = firstLine.includes(';') ? ';' : ','
  const rawHeaders = firstLine.split(delimiter).map(h => h.trim().toLowerCase().replace(/['"]+/g, ''))

  // Find column indices by flexible keyword matching
  const findIdx = (keywords: string[]) => rawHeaders.findIndex(h => keywords.some(k => h.includes(k)))

  const ghiIdx = findIdx(['ghi', 'irradiance', 'solar', 'poa', 'rad'])
  const windIdx = findIdx(['wind_speed', 'windspeed', 'wind', 'speed', 'ws', 'p_wind'])
  const tempIdx = findIdx(['temperature', 'temp', 'degc'])
  const dirIdx = findIdx(['wind_dir', 'winddir', 'direction', 'dir', 'azimuth'])
  const timeIdx = findIdx(['timestamp', 'time', 'date', 'datetime', 'hour'])

  const rows: ImportedRow[] = []
  let totalSolarKW = 0
  let totalWindKW = 0
  let totalKW = 0
  let peakKW = 0

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(delimiter).map(c => c.trim().replace(/['"]+/g, ''))
    if (cols.length < 2) continue

    const parseNum = (idx: number, fallback: number) => {
      if (idx === -1 || idx >= cols.length) return fallback
      const val = parseFloat(cols[idx])
      return isNaN(val) ? fallback : val
    }

    const timestamp = timeIdx !== -1 && cols[timeIdx] ? cols[timeIdx] : `Hour ${i}`
    const rawGhi = parseNum(ghiIdx, 350)
    // Normalize GHI if it was given in kW/m² instead of W/m²
    const ghi = rawGhi < 3.0 && rawGhi > 0 ? rawGhi * 1000 : rawGhi
    const windSpeed = parseNum(windIdx, 7.0)
    const temperature = parseNum(tempIdx, 18.0)
    const windDirection = parseNum(dirIdx, 180.0)

    const prediction = predictYieldLocal({
      ghi,
      windSpeed,
      temperature,
      windDirection,
      humidity: 50,
      lagPower: 5,
    })

    totalSolarKW += prediction.solar
    totalWindKW += prediction.wind
    totalKW += prediction.total
    if (prediction.total > peakKW) peakKW = prediction.total

    rows.push({
      timestamp,
      ghi: Math.round(ghi),
      windSpeed: Number(windSpeed.toFixed(1)),
      temperature: Number(temperature.toFixed(1)),
      windDirection: Math.round(windDirection),
      predictedSolar: prediction.solar,
      predictedWind: prediction.wind,
      predictedTotal: prediction.total,
    })
  }

  const count = rows.length || 1
  return {
    fileName,
    rows,
    totalEnergyKWh: Number(totalKW.toFixed(1)),
    peakPowerKW: Number(peakKW.toFixed(2)),
    avgSolarKW: Number((totalSolarKW / count).toFixed(2)),
    avgWindKW: Number((totalWindKW / count).toFixed(2)),
    avgTotalKW: Number((totalKW / count).toFixed(2)),
    avgCapacityFactor: Number(((totalKW / count / 18.0) * 100).toFixed(1)),
  }
}

// Generate a ready-to-use sample SOLETE CSV dataset
export function generateSampleCsvContent(): string {
  const headers = 'timestamp,irradiance_ghi_w_m2,wind_speed_m_s,temperature_degc,wind_direction_deg\n'
  const rows: string[] = []
  for (let h = 0; h < 24; h++) {
    const timeStr = `2024-06-15 ${String(h).padStart(2, '0')}:00:00`
    const daylight = Math.max(0, Math.sin(((h - 5) / 14) * Math.PI))
    const ghi = Math.round(daylight * (780 + Math.sin(h) * 50))
    const windSpeed = (5.5 + Math.sin(h * 0.5) * 3.8 + (h > 17 ? 4.2 : 0)).toFixed(1)
    const temp = (14 + daylight * 11).toFixed(1)
    const dir = Math.round(180 + Math.sin(h * 0.4) * 45)
    rows.push(`${timeStr},${ghi},${windSpeed},${temp},${dir}`)
  }
  return headers + rows.join('\n')
}

// Browser File Downloader
export function triggerFileDownload(filename: string, content: string, mimeType = 'text/csv;charset=utf-8;') {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.setAttribute('href', url)
  link.setAttribute('download', filename)
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}
