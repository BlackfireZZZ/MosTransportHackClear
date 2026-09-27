import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  componentLabel,
  formatKm,
  groupRoutesByComponent,
} from "@/features/tram-network/lib/network"
import type { TramRoute } from "@/features/tram-network/types"

export const ALL_ROUTES = "__all__"

interface RouteFilterProps {
  readonly routes: readonly TramRoute[]
  readonly value: string | null
  readonly onChange: (ref: string | null) => void
}

export function RouteFilter({ routes, value, onChange }: RouteFilterProps) {
  const groups = groupRoutesByComponent(routes)

  return (
    <div className="filter-field">
      <label htmlFor="tram-route-select">Маршрут</label>
      <Select
        value={value ?? ALL_ROUTES}
        onValueChange={(next) => onChange(next === ALL_ROUTES ? null : next)}
      >
        <SelectTrigger id="tram-route-select" className="tram-route-trigger">
          <SelectValue placeholder="Вся сеть" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL_ROUTES}>Вся сеть · {routes.length} маршрутов</SelectItem>
          {groups.map((group) => (
            <SelectGroup key={group.component}>
              <SelectLabel>{componentLabel(group.component)}</SelectLabel>
              {group.routes.map((route) => (
                <SelectItem key={route.ref} value={route.ref}>
                  {route.ref} · {route.stop_count} ост. · {formatKm(route.length_m)}
                </SelectItem>
              ))}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
