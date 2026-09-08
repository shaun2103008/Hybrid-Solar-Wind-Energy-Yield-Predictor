export const USE_LIVE_BACKEND = false // Toggle to true when FastAPI backend is deployed

export type PredictionInputs = { ghi: number; windSpeed: number; temperature: number; windDirection: number; humidity: number; lagPower: number }
export type Prediction = { solar: number; wind: number; total: number; ratio: number; capacityFactor: number; forecast: ForecastPoint[] }
export type ForecastPoint = { hour: string; solar: number; wind: number; total: number; ghi: number; windSpeed: number }

export const presets: Record<string, PredictionInputs> = {
  'Sunny Calm Day': { ghi: 820, windSpeed: 4.2, temperature: 24, windDirection: 145, humidity: 42, lagPower: 4.4 },
  'Stormy Windy Night': { ghi: 0, windSpeed: 18.5, temperature: 9, windDirection: 285, humidity: 88, lagPower: 6.9 },
  'Cloudy Afternoon': { ghi: 280, windSpeed: 8.8, temperature: 17, windDirection: 210, humidity: 74, lagPower: 3.2 },
  'Peak Winter Solar': { ghi: 640, windSpeed: 6.5, temperature: 3, windDirection: 165, humidity: 55, lagPower: 3.9 },
}

export function predictYield(input: PredictionInputs): Prediction {
  const solar = Math.min(11, Math.max(0, input.ghi / 1000 * 8.4 * (1 - Math.max(0, input.temperature - 25) * 0.004)))
  const wind = Math.min(11, Math.max(0, Math.pow(input.windSpeed / 12, 2.35) * 5.4 + Math.max(0, input.windSpeed - 12) * 0.12))
  const total = Math.min(11, solar + wind * 0.62 + input.lagPower * 0.12)
  const ratio = total ? solar / total * 100 : 0
  const capacityFactor = total / 11 * 100
  const forecast = Array.from({ length: 24 }, (_, hour) => {
    const daylight = Math.max(0, Math.sin((hour - 6) / 13 * Math.PI))
    const solarPoint = Math.max(0, solar * (0.22 + daylight * 0.88) * (0.88 + Math.sin(hour * 1.7) * 0.05))
    const windPoint = Math.max(0, wind * (0.7 + Math.sin(hour * 0.68 + input.windDirection / 80) * 0.18 + Math.sin(hour * 1.9) * 0.08))
    return { hour: `${String(hour).padStart(2, '0')}:00`, solar: Number(solarPoint.toFixed(2)), wind: Number(windPoint.toFixed(2)), total: Number((solarPoint + windPoint * 0.62).toFixed(2)), ghi: Math.round(input.ghi * daylight), windSpeed: Number((input.windSpeed * (0.9 + Math.sin(hour * 0.68) * 0.12)).toFixed(1)) }
  })
  return { solar: Number(solar.toFixed(2)), wind: Number(wind.toFixed(2)), total: Number(total.toFixed(2)), ratio: Number(ratio.toFixed(1)), capacityFactor: Number(capacityFactor.toFixed(1)), forecast }
}

export function exportPrediction(prediction: Prediction, format: 'csv' | 'json') {
  if (format === 'json') return JSON.stringify(prediction, null, 2)
  return ['hour,solar_kw,wind_kw,total_kw,ghi,wind_speed', ...prediction.forecast.map((p) => `${p.hour},${p.solar},${p.wind},${p.total},${p.ghi},${p.windSpeed}`)].join('\n')
}
