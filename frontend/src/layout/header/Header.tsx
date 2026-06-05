"use client";
import Link from "next/link";
import Image from "next/image";
import { useState } from "react";

import { Menu } from "lucide-react";
import { usePathname } from "next/navigation";
import { MenuItem } from "../Sidebar";
import { Button } from "@/src/components/ui/button";
import { ThemeToggleButton } from "@/src/components/ui/ThemeToggleButton";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/src/components/ui/dropdown-menu";

export const MENU_ITEMS: MenuItem[] = [
  { label: "BÁO CÁO", href: "#content-report" },
  { label: "CHIẾN DỊCH", href: "#campaigns" },
  { label: "TUYÊN BỐ MỚI NHẤT", href: "#latest-claims" },
];

export default function Header() {
  const pathname = usePathname();
  const currentItem = MENU_ITEMS.find((item) => item.href === pathname);
  const [selectedMenu, setSelectedMenu] = useState<string>(
    currentItem ? currentItem.label : MENU_ITEMS[0].label,
  );

  return (
    <div className="fixed top-0 left-0 right-0 z-50 w-full h-16 bg-background border-b px-1 sm:px-2 md:px-6">
      <div className="h-full flex items-center justify-between px-4 md:px-6 w-full max-w-8xl mx-auto">
        <div className="flex items-center gap-15 w-full">
          <div className="h-full flex items-center justify-start gap-2.5">
            <Link href="/" className="flex items-center">
              <Image src="/LOGO.png" alt="Biểu trưng" width={35} height={42} />
            </Link>
            <h1 className="text-2xl font-semibold hidden md:flex bg-clip-text text-transparent bg-[image:var(--FN-Gradient-2)]">
              Fake Finance Detector
            </h1>
          </div>
          <div className="h-full hidden lg:flex items-center justify-center ">
            <div className="hidden lg:flex items-center gap-12 mt-1.5">
              {MENU_ITEMS.map((item) => {
                const isSelected = selectedMenu === item.label;
                return (
                  <Link
                    key={item.href}
                    href={item.comingSoon ? "#" : item.href}
                    onClick={() =>
                      !item.comingSoon && setSelectedMenu(item.label)
                    }
                    className={`mb-1 py-3 rounded-sm text-md transition-colors flex items-center gap-3 ${
                      isSelected
                        ? "text-primary font-semibold"
                        : "text-primary hover:text-foreground"
                    } ${item.comingSoon ? "opacity-80" : ""}`}
                    style={
                      isSelected
                        ? { textShadow: "0 0 10px var(--shadow)" }
                        : undefined
                    }
                  >
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        </div>
        <div className="h-full flex items-center justify-end gap-2">
          <ThemeToggleButton />

          <div className="lg:hidden">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size="lg" variant="contained1" aria-label="Mở menu">
                  <Menu className="w-5 h-5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                {MENU_ITEMS.map((item: MenuItem) => {
                  const isSelected = selectedMenu === item.label;
                  return (
                    <Link
                      key={item.href}
                      href={item.comingSoon ? "#" : item.href}
                      onClick={() =>
                        !item.comingSoon && setSelectedMenu(item.label)
                      }
                    >
                      <DropdownMenuItem
                        className={`cursor-pointer ${
                          item.comingSoon ? "opacity-80" : ""
                        }`}
                      >
                        {item.icon && (
                          <span className="shrink-0">{item.icon}</span>
                        )}
                        <span
                          style={
                            isSelected
                              ? {
                                  background: "var(--FN-Gradient-1)",
                                  WebkitBackgroundClip: "text",
                                  WebkitTextFillColor: "transparent",
                                  backgroundClip: "text",
                                  fontWeight: 600,
                                }
                              : undefined
                          }
                        >
                          {item.label}
                        </span>
                      </DropdownMenuItem>
                    </Link>
                  );
                })}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </div>
  );
}
