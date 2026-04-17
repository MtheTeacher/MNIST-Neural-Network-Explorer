import { useEffect, useRef, useState } from 'react';

/**
 * Custom hook to measure a container's dimensions for Recharts.
 * This is more robust than ResponsiveContainer in some environments.
 */
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
