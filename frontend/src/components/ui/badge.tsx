import * as React from "react"

import { cn } from "@/lib/utils"

export function Badge({ className, ...props }: React.ComponentProps<"span">) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm border border-border bg-muted px-2 py-1 text-xs font-medium text-foreground",
        className,
      )}
      {...props}
    />
  )
}
