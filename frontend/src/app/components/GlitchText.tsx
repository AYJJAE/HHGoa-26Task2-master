"use client";

import React from "react";
import "./GlitchText.css";

export interface GlitchTextProps {
  children: React.ReactNode;
  speed?: number;
  enableShadows?: boolean;
  enableOnHover?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

export default function GlitchText({
  children,
  speed = 1,
  enableShadows = true,
  enableOnHover = true,
  className = "",
  style = {},
}: GlitchTextProps) {
  const textContent = typeof children === "string" ? children : React.Children.toArray(children).join("");

  const inlineStyle: React.CSSProperties = {
    ...style,
    ["--glitch-speed" as any]: speed,
  };

  const textClasses = [
    "glitch-text",
    enableShadows ? "enable-shadows" : "",
    enableOnHover ? "hover-only" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <span className="glitch-container" style={inlineStyle}>
      <span className={textClasses} data-text={textContent}>
        {children}
      </span>
    </span>
  );
}
