type GradientBorderCardV1Props = {
  height?: string;
  width?: string;
  padding?: string;
  background?: string;
  className?: string;
  children?: React.ReactNode;
};

const GradientBorderCardV1 = ({
  children,
  height,
  width,
  padding = "10px",
  background,
}: GradientBorderCardV1Props) => {
  return (
    <div className="relative" style={{ height, width }}>
      {/* Gradient border wrapper */}
      <div
        className="absolute inset-0 rounded-[20px] overflow-hidden"
        style={{
          backgroundColor: background || "var(--background)",
        }}
      >
        {/* Top-left corner */}
        <div
          className="absolute top-0 left-0 w-[30px] h-[30px]"
          style={{
            background:
              "linear-gradient(135deg, #FFFFFF 0%, #4DE5C5 30%, #6AABFB 70%, #9133FF 100%)",
            clipPath: "polygon(0 0, 100% 0, 0 100%)",
          }}
        >
          <div
            className="w-full h-full mt-[1.5px] ml-[1.5px] rounded-tl-[20px]"
            style={{
              backgroundColor: background || "var(--background)",
            }}
          ></div>
        </div>

        {/* Bottom-right corner */}
        <div
          className="absolute bottom-0 right-0 w-[30px] h-[30px]"
          style={{
            background:
              "linear-gradient(-45deg, #FFFFFF 0%, #4DE5C5 30%, #6AABFB 70%, #9133FF 100%)",
            clipPath: "polygon(100% 0, 100% 100%, 0 100%)",
          }}
        >
          <div
            className="absolute w-full h-full bottom-[1.5px] right-[1.5px] rounded-br-[20px]"
            style={{
              backgroundColor: background || "var(--background)",
            }}
          ></div>
        </div>
      </div>

      {/* Content with padding */}
      <div
        className="relative z-10 h-full w-full rounded-[20px]"
        style={{ padding }}
      >
        {children}
      </div>
    </div>
  );
};

export default GradientBorderCardV1;
