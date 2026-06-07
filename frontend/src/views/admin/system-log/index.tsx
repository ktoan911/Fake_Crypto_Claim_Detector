import IntroAppPage from "@/src/components/IntroAppPage";
import SystemLogPanel from "./SystemLog";

export default function SystemLogSection() {
  return (
    <>
      <IntroAppPage
        title="System Logs"
        description="Raw server output log"
      ></IntroAppPage>
      <SystemLogPanel />
    </>
  );
}
