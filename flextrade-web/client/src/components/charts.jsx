/* Shared Recharts wrappers, styled with the design tokens.
   Rules followed (dataviz method): one y-axis per chart, thin marks,
   2px lines, recessive grid, legend for >=2 series, tooltips on hover. */
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line,
  LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

const axis = { stroke: "var(--baseline)", tick: { fill: "var(--muted)", fontSize: 11 }, tickLine: false };
const grid = { stroke: "var(--grid)", vertical: false };
export const tooltipStyle = {
  contentStyle: {
    background: "var(--surface)", border: "1px solid var(--border)",
    borderRadius: 8, fontSize: 12, color: "var(--ink)",
  },
  labelStyle: { color: "var(--ink-2)", fontWeight: 600 },
};
const legendStyle = { wrapperStyle: { fontSize: 12 } };

export function TimeSeries({ data, series, height = 260, xKey = "ts", yLabel, refLines = [] }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 4 }}>
        <CartesianGrid {...grid} />
        <XAxis dataKey={xKey} {...axis} minTickGap={40} />
        <YAxis {...axis} width={54}
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 11 } : undefined} />
        <Tooltip {...tooltipStyle} />
        {series.length > 1 && <Legend {...legendStyle} />}
        {refLines.map((r, i) => (
          <ReferenceLine key={i} y={r.y} stroke={r.color || "var(--critical)"}
            strokeDasharray="4 3" label={{ value: r.label, fill: "var(--muted)", fontSize: 10 }} />
        ))}
        {series.map((s) =>
          s.type === "area" ? (
            <Area key={s.key} dataKey={s.key} name={s.name} stroke={s.color}
              fill={s.fill || s.color} fillOpacity={s.fillOpacity ?? 0.12}
              strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
          ) : s.type === "bar" ? (
            <Bar key={s.key} dataKey={s.key} name={s.name} fill={s.color}
              isAnimationActive={false} radius={[3, 3, 0, 0]} />
          ) : (
            <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color}
              strokeWidth={2} dot={false} isAnimationActive={false} connectNulls
              strokeDasharray={s.dash} />
          )
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* Fan chart: q10-q90 band + median. Band drawn as stacked areas (invisible
   base up to q10, visible band of q90-q10). */
export function FanChart({ data, height = 280, xKey = "ts", medianKey = "q50", loKey = "q10", hiKey = "q90", extra = [] }) {
  const prepared = data.map((d) => ({ ...d, _band: d[hiKey] != null && d[loKey] != null ? d[hiKey] - d[loKey] : null }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={prepared} margin={{ top: 6, right: 8, bottom: 0, left: 4 }}>
        <CartesianGrid {...grid} />
        <XAxis dataKey={xKey} {...axis} minTickGap={40} />
        <YAxis {...axis} width={56} />
        <Tooltip {...tooltipStyle} />
        <Legend {...legendStyle} />
        <Area dataKey={loKey} stackId="band" stroke="none" fill="transparent"
          name="P10" isAnimationActive={false} legendType="none" tooltipType="none" />
        <Area dataKey="_band" stackId="band" stroke="none" fill="var(--band)"
          name="P10–P90 band" isAnimationActive={false} />
        <Line dataKey={medianKey} name="P50 (median)" stroke="var(--s1)"
          strokeWidth={2} dot={false} isAnimationActive={false} />
        {extra.map((s) => (
          <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color}
            strokeWidth={2} strokeDasharray={s.dash} dot={false} isAnimationActive={false} />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function HBar({ data, height = 300, cat = "name", val = "value", color = "var(--s1)", valLabel }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 30, bottom: 0, left: 10 }}>
        <CartesianGrid stroke="var(--grid)" horizontal={false} />
        <XAxis type="number" {...axis} />
        <YAxis type="category" dataKey={cat} {...axis} width={110} />
        <Tooltip {...tooltipStyle} />
        <Bar dataKey={val} name={valLabel || val} fill={color} radius={[0, 4, 4, 0]}
          isAnimationActive={false} barSize={16} />
      </BarChart>
    </ResponsiveContainer>
  );
}
