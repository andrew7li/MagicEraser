import styles from "./first.module.scss";

import { RiUploadLine } from "react-icons/ri";

type FirstProps = {
  setWorkflow: (newWorkflow: number) => void;
  setFile: (file: File | null) => void;
};

export default function First(props: FirstProps) {
  const { setWorkflow, setFile } = props;

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      setFile(files[0]);
    }
    setWorkflow(1);
  };

  return (
    <div className={styles.imageContainer}>
      <div className={styles.imageDemoContainer}>
        <video autoPlay muted loop preload="none">
          <source src="/2160p.MOV" type="video/mp4" />
          <track
            src="/path/to/captions.vtt"
            kind="subtitles"
            srcLang="en"
            label="English"
          />
          Your browser does not support the video tag.
        </video>
      </div>
      <div className={styles.imageUploadContainer}>
        <input
          type="file"
          id="inputFile"
          style={{ display: "none" }} // Hide the input but keep it functional
          accept="image/jpeg, image/png, image/webp, image/bmp" // Specify accepted file types
          onChange={handleFileUpload} // Implement a function to handle file uploads
        />
        <div className={styles.uploadImageDragAndDropContainer}>
          <label htmlFor="inputFile" className={styles.uploadImageTextLabel}>
            <div>
              <RiUploadLine />
            </div>
            <ul>
              <li>Drag or drop images here</li>
              <li>JPG, PNG, JPEG, WEBP, BMP· Max size 5Mb</li>
            </ul>
          </label>

          <p>
            *All files are stored privately & encrypted. Only you will see them.
          </p>
        </div>

        <label htmlFor="inputFile" className={styles.uploadImageButtonLabel}>
          <div className={styles.uploadButton}>Upload</div>
        </label>

        <div className={styles.sampleImagesContainer}>
          <p>Try one of these now!</p>
          <ul>
            <li>
              <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
            </li>
            <li>
              <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
            </li>
            <li>
              <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
            </li>
            <li>
              <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
            </li>
            <li>
              <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
            </li>
            <li>
              <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
