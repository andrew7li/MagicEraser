import styles from "./fourth.module.scss";

type FourthProps = {
  finalOutputUrl: string | undefined;
  setWorkflow: (newWorkflow: number) => void;
};

export default function Fourth(props: FourthProps) {
  const { finalOutputUrl, setWorkflow } = props;

  /**
   * Handler for when download button is clicked.
   */
  const handleDownloadButtonClick = () => {};

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
