import IntroAppPage from "@/src/components/IntroAppPage";
import CardItem from "./CardItem";
import TableDetect from "./TableDetect";

export default function Campaigns() {
  return (
    <div>
      <IntroAppPage
        title="Chiến Dịch"
        description="Các chủ đề tuyên bố được gom cụm theo mức độ lan truyền và mức độ rủi ro."
      />
      <div className="mt-10">
        <CardItem />
      </div>
      <div id="latest-claims" className="mt-10">
        <TableDetect />
      </div>
    </div>
  );
}
