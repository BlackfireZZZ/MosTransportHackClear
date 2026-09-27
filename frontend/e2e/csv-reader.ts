export function readCsv(csv: string): Record<string, string>[] {
  const records: string[][] = []
  let row: string[] = [], field = "", quoted = false
  const input = csv.replace(/^\uFEFF/, "")
  for (let i = 0; i < input.length; i++) {
    const char = input[i]
    if (char === '"') {
      if (quoted && input[i + 1] === '"') { field += '"'; i++ }
      else quoted = !quoted
    } else if (!quoted && char === ",") { row.push(field); field = "" }
    else if (!quoted && char === "\n") {
      row.push(field.replace(/\r$/, "")); records.push(row); row = []; field = ""
    } else field += char
  }
  const [headers, ...values] = records
  return values.map((cells) => Object.fromEntries(headers.map((header, index) => [header, cells[index]])))
}
