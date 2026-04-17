# Recharts Best Practices & Implementation Guide

This document explains exactly how we implemented the charts in this application, what problems we encountered with Recharts, what failed, and our robust, working pattern. **Please read this before modifying Rechart implementations in this application.**

## The Problems Encountered

1. **`ResponsiveContainer` Silently Failing:**
   When wrapping `<LineChart>` inside Recharts' built-in `<ResponsiveContainer>`, the charts would heavily artifact or collapse to a 0x0 size, especially when deeply nested inside CSS Grid or Flex containers within an `iframe` and sandboxed environments.
   
2. **Animation Glitches on Real-Time Streams:**
   Because our data streams incrementally per epoch, Recharts' default transition animations couldn't keep up. This caused lines to glitch, draw backwards, or vanish completely during training updates.

3. **`NaN` and `Infinity` Silently Breaking Charts:**
   When our training loop produced invalid values (like `NaN` or `Infinity` resulting from zero-divisions in learning rate schedulers), Recharts choked on them. It does not handle invalid SVG coordinates well, which led to the entire line failing to render or the Y-Axis ranges breaking.

---

## 1. The Real-Time Responsive Pattern (Our Fix for #1)

Instead of relying on `<ResponsiveContainer>`, we implemented a custom `ResizeObserver` pattern via `useChartDimensions`. 

### `useChartDimensions.ts`
We measure the exact `clientWidth` and `clientHeight` of the container to pass explicit numeric values to the chart.

```typescript
import { useEffect, useRef, useState } from 'react';

export function useChartDimensions() {
    const ref = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

    useEffect(() => {
        if (!ref.current) return;

        const observer = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (entry && entry.contentRect) {
                const w = entry.contentRect.width;
                const h = entry.contentRect.height;
                
                // Ensure we only set valid, finite numbers
                setDimensions({
                    width: isFinite(w) ? Math.floor(w) : 0,
                    height: isFinite(h) ? Math.floor(h) : 0
                });
            }
        });

        observer.observe(ref.current);
        return () => observer.disconnect();
    }, []);

    return [ref, dimensions] as const;
}
```

### `ChartWrapper` Component
We use a wrapper component taking a render prop. This ensures the chart only renders when valid dimensions are > 0. 

```tsx
const ChartWrapper = ({ children }: { children: (width: number, height: number) => React.ReactNode }) => {
    const [ref, size] = useChartDimensions();
    return (
        <div ref={ref} className="w-full h-[350px] relative">
            {size.width > 0 && size.height > 0 ? children(size.width, size.height) : (
                <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                    Initializing chart...
                </div>
            )}
        </div>
    );
};
```

**Usage:**
```tsx
<ChartWrapper>
    {(width, height) => (
        <LineChart width={width} height={height} data={sanitizedData}>
            {/* Chart components... */}
        </LineChart>
    )}
</ChartWrapper>
```

---

## 2. Animation Attributes (Our Fix for #2)

When plotting live, high-frequency updates, **always disable Recharts animations** on the graphic components. It ensures a stable line is constantly painted without internal React transition delays.

```tsx
<Line 
  yAxisId="loss" 
  type="monotone" 
  dataKey="loss" 
  isAnimationActive={false} // <-- CRITICAL for real-time rendering
  connectNulls={true}       // <-- Helpful when some epochs miss val_loss
/>
```

---

## 3. Data Sanitization (Our Fix for #3)

If **even a single data point** passes mathematical edge cases yielding `NaN`, `Infinity`, or `-Infinity`, the path drawing algorithm will break.

**Best Practice:**
Always sanitize your dataset immediately before feeding it into the Chart data prop. We built a `sanitize(num)` helper that coerces broken math to `0`.

```typescript
const sanitize = (val: any) => {
    if (typeof val === 'number') {
        if (isNaN(val) || !isFinite(val)) return 0; // Replace invalid numbers
        return val;
    }
    return val;
};

// In render:
const safeLog = trainingLog.map(entry => ({
    ...entry,
    loss: sanitize(entry.loss),
    lr: sanitize(entry.lr),
    acc: sanitize(entry.acc),
    val_loss: sanitize(entry.val_loss),
    val_acc: sanitize(entry.val_acc),
}));

// Pass safeLog to Recharts
<LineChart data={safeLog} ...>
```

## Summary Checklist
If charts stop rendering in the future:
1. Ensure `isAnimationActive={false}` is on all `<Line>` elements.
2. Ensure you are using `<ChartWrapper>` + `useChartDimensions` and NOT `ResponsiveContainer`.
3. Log the raw array being fed to `data={}`. Search for `null` in places that shouldn't be null, `NaN`, or `Infinity`. 
4. Check math functions upstream (especially Learning Rate schedulers and division by zero!).
