"use client";

import React, { useRef, useEffect, useState } from "react";

export interface AnimatedContentProps {
  children: React.ReactNode;
  distance?: number;
  direction?: "vertical" | "horizontal";
  reverse?: boolean;
  duration?: number;
  ease?: string;
  initialOpacity?: number;
  animateOpacity?: boolean;
  scale?: number;
  threshold?: number;
  delay?: number;
  className?: string;
  style?: React.CSSProperties;
}

export default function AnimatedContent({
  children,
  distance = 150,
  direction = "horizontal",
  reverse = false,
  duration = 1.2,
  ease = "bounce.out",
  initialOpacity = 0.2,
  animateOpacity = true,
  scale = 1.1,
  threshold = 0.2,
  delay = 0.3,
  className = "",
  style = {},
}: AnimatedContentProps) {
  const [inView, setInView] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Use IntersectionObserver with fallback for immediate trigger if already in viewport
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.unobserve(entry.target);
        }
      },
      { threshold: Math.min(Math.max(threshold, 0), 1) }
    );

    observer.observe(el);

    // Initial check in case it's in viewport on load
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      setInView(true);
    }

    return () => observer.disconnect();
  }, [threshold]);

  // Easing mapping for CSS transitions
  const getCubicBezier = (easing: string): string => {
    switch (easing) {
      case "bounce.out":
      case "bounce":
        // Characteristic bounce curve
        return "cubic-bezier(0.34, 1.56, 0.64, 1)";
      case "elastic.out":
        return "cubic-bezier(0.68, -0.6, 0.32, 1.6)";
      case "power3.out":
      case "ease-out":
        return "cubic-bezier(0.215, 0.61, 0.355, 1)";
      case "power2.out":
        return "cubic-bezier(0.25, 0.46, 0.45, 0.94)";
      default:
        return easing.includes("cubic-bezier") ? easing : "cubic-bezier(0.34, 1.56, 0.64, 1)";
    }
  };

  const getInitialTransform = (): string => {
    const multiplier = reverse ? -1 : 1;
    const x = direction === "horizontal" ? distance * multiplier : 0;
    const y = direction === "vertical" ? distance * multiplier : 0;
    const s = scale !== 1 ? `scale(${scale})` : "";
    return `translate3d(${x}px, ${y}px, 0) ${s}`.trim();
  };

  const animationStyle: React.CSSProperties = {
    opacity: inView ? 1 : animateOpacity ? initialOpacity : 1,
    transform: inView ? "translate3d(0, 0, 0) scale(1)" : getInitialTransform(),
    transition: `opacity ${duration}s ${getCubicBezier(ease)} ${delay}s, transform ${duration}s ${getCubicBezier(ease)} ${delay}s`,
    willChange: "opacity, transform",
    ...style,
  };

  return (
    <div ref={ref} style={animationStyle} className={className}>
      {children}
    </div>
  );
}
