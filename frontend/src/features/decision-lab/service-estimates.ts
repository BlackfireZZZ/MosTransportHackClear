import snapshot from "./service-estimates.json"

export interface ServiceEstimate {
  cycleMinutes: number
  cycleP10: number
  cycleP90: number
  pairedCycles: number
  departuresByHour: number[]
  sourceUrl: string
  sourceSha256: string
  captureDate: string
}

type Profile = (typeof snapshot.profiles)[keyof typeof snapshot.profiles][number]

function profileFor(route: string, day: string): Profile | null {
  const profiles = snapshot.profiles[route as keyof typeof snapshot.profiles]
  if (!profiles) return null
  const weekday = (new Date(`${day}T12:00:00+03:00`).getUTCDay() + 6) % 7
  const matches = profiles.filter((profile) =>
    profile.weekdays.includes(weekday) && profile.validFrom <= day.replaceAll("-", "") && profile.validTo >= day.replaceAll("-", ""))
  return matches.length === 1 ? matches[0] : null
}

export function serviceEstimate(route: string, day: string): ServiceEstimate | null {
  const today = profileFor(route, day)
  if (!today) return null
  const prior = new Date(`${day}T12:00:00+03:00`)
  prior.setUTCDate(prior.getUTCDate() - 1)
  const previousDay = prior.toISOString().slice(0, 10)
  const yesterday = profileFor(route, previousDay)
  if (!yesterday) return null
  return {
    cycleMinutes: today.cycleMinutes,
    cycleP10: today.cycleP10,
    cycleP90: today.cycleP90,
    pairedCycles: today.pairedCycles,
    departuresByHour: today.departureStartsByServiceHour.slice(0, 24).map((value, hour) => value + yesterday.departureStartsByServiceHour[hour + 24]),
    sourceUrl: snapshot.sourceUrl,
    sourceSha256: snapshot.sourceSha256,
    captureDate: snapshot.captureDate,
  }
}
