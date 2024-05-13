import styles from "./fourth.module.scss";
import { saveAs } from "file-saver";

type FourthProps = {
  finalOutputUrl: string | undefined;
  setWorkflow: (newWorkflow: number) => void;
  fileName: string;
};

export default function Fourth(props: FourthProps) {
  const { finalOutputUrl, setWorkflow, fileName } = props;

  /**
   * Handler for when download button is clicked.
   */
  const handleDownloadButtonClick = () => {
    saveAs(finalOutputUrl, fileName);
  };

  /**
   * Handler for when upload new image is clicked.
   */
  const handleUploadNewImageButtonClick = () => {
    setWorkflow(0);
  };

  return !finalOutputUrl ? (
    <div>Error! No final output url found!</div>
  ) : (
    <>
      <div className={styles.leftContainer}>
        <img src={finalOutputUrl} />
      </div>
      <div className={styles.rightContainer}>
        <p>Thanks for using Magic Eraser! Download image here.</p>
        <div id={styles.button} onClick={handleDownloadButtonClick}>
          Download
        </div>
        <div id={styles.button} onClick={handleUploadNewImageButtonClick}>
          Upload New Image
        </div>
      </div>
    </>
  );
}
