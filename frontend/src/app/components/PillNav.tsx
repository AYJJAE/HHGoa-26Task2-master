"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import "./PillNav.css";

export interface NavItem {
  label: string;
  href: string;
}

export interface PillNavProps {
  items: NavItem[];
  activeHref?: string;
  className?: string;
  ease?: string;
  baseColor?: string;
  pillColor?: string;
  hoveredPillTextColor?: string;
  pillTextColor?: string;
  logo?: string;
  logoAlt?: string;
}

export default function PillNav({
  items,
  activeHref = "",
  className = "",
  ease = "power2.easeOut",
  baseColor = "rgba(6, 26, 18, 0.85)",
  pillColor = "#FFE600",
  hoveredPillTextColor = "#061A12",
  pillTextColor = "#FFFFFF",
}: PillNavProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [activeIndex, setActiveIndex] = useState<number | null>(
    items.findIndex((item) => item.href === activeHref) !== -1
      ? items.findIndex((item) => item.href === activeHref)
      : null
  );
  
  const navRef = useRef<HTMLDivElement>(null);
  const [pillStyle, setPillStyle] = useState<React.CSSProperties>({});
  
  useEffect(() => {
    if (navRef.current) {
      const targetIndex = hoveredIndex !== null ? hoveredIndex : activeIndex;
      if (targetIndex !== null) {
        const itemEls = navRef.current.querySelectorAll(".pill-nav-item");
        const targetEl = itemEls[targetIndex] as HTMLElement;
        if (targetEl) {
          setPillStyle({
            width: targetEl.offsetWidth,
            transform: `translateX(${targetEl.offsetLeft}px)`,
            opacity: 1,
            backgroundColor: pillColor,
          });
        }
      } else {
        setPillStyle({ opacity: 0 });
      }
    }
  }, [hoveredIndex, activeIndex, pillColor]);

  return (
    <div
      className={`pill-nav-container ${className}`}
      style={{ backgroundColor: baseColor }}
      ref={navRef}
      onMouseLeave={() => setHoveredIndex(null)}
    >
      <div
        className="pill-nav-indicator"
        style={{
          ...pillStyle,
          transition: `all 0.3s cubic-bezier(0.25, 1, 0.5, 1)`,
        }}
      />
      {items.map((item, index) => {
        const isExternal = item.href.startsWith("http");
        const isHovered = hoveredIndex === index;
        const isActive = activeIndex === index && hoveredIndex === null;

        const textColor = isHovered || isActive ? hoveredPillTextColor : pillTextColor;

        const content = (
          <span className="pill-nav-text" style={{ color: textColor }}>
            {item.label}
          </span>
        );

        return (
          <div
            key={item.label}
            className="pill-nav-item"
            onMouseEnter={() => setHoveredIndex(index)}
            onClick={() => setActiveIndex(index)}
          >
            {isExternal ? (
              <a href={item.href} target="_blank" rel="noopener noreferrer" className="pill-nav-link">
                {content}
              </a>
            ) : (
              <Link href={item.href} className="pill-nav-link">
                {content}
              </Link>
            )}
          </div>
        );
      })}
    </div>
  );
}
