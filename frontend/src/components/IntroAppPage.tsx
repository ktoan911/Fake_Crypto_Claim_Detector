"use client";

type IntroAppPageProps = {
  title: string;
  description: string;
};

const IntroAppPage = ({ title, description }: IntroAppPageProps) => {
  return (
    <div className="flex flex-col items-start justify-center pl-0 bg">
      <h1 className="text-4xl animate-intro-title [text-shadow:0_2px_12px_var(--text-shadow)]">
        {title}
      </h1>
      <p className="text-[16px] animate-intro-description opacity-76">
        {description}
      </p>
    </div>
  );
};

export default IntroAppPage;
