import styles from "./fourth.module.scss";
import { saveAs } from "file-saver";

type FourthProps = {
  finalOutputUrl: string | undefined;
  setWorkflow: (newWorkflow: number) => void;
  fileName: string | undefined;
};

export default function Fourth(props: FourthProps) {
  const { finalOutputUrl, setWorkflow, fileName } = props;

  /**
   * Handler for when download button is clicked.
   */
  const handleDownloadButtonClick = () => {
    if (!finalOutputUrl) {
      console.error("Final output URL not defined", finalOutputUrl);
      return;
    }
    const newFileName = fileName ?? "dummy-name.jpeg";
    saveAs(finalOutputUrl, newFileName);
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
        <p>
          Thanks for using Magic Eraser! Feel free to download the image or
          upload a new image!
        </p>
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
