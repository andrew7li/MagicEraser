import styles from "./first.module.scss";

import { RiUploadLine } from "react-icons/ri";

type FirstProps = {
  setWorkflow: (newWorkflow: number) => void;
  setFile: (file: File | null) => void;
};

const imgUrls = [
  "sample1.jpg",
  "sample2.jpeg",
  "sample3.jpeg",
  "sample4.avif",
  "sample5.jpeg",
  "sample6.jpeg",
];

export default function First(props: FirstProps) {
  const { setWorkflow, setFile } = props;

  /**
   * Function to handle file upload.
   */
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      setFile(files[0]);
    }
    setWorkflow(1);
  };

  /**
   * Function to handle image click.
   */
  const handleImageClick = (idx: number) => {
    const imgUrl = imgUrls[idx];

    fetch(imgUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.blob(); // Convert the response to a blob.
      })
      .then((blob) => {
        const file = new File([blob], imgUrl, { type: blob.type }); // Creates a file with the correct MIME type.
        setFile(file); // Update the state with the new file.
      })
      .catch((error) => {
        console.error("Error fetching the image:", error);
      });

    setWorkflow(1);
  };

  return (
    <>
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
          style={{ display: "none" }}
          accept="image/jpeg, image/png, image/webp, image/bmp" // Specify accepted file types
          onChange={handleFileUpload}
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
            {imgUrls.map((ele, idx) => (
              <li key={ele}>
                <img
                  src={ele}
                  alt="description"
                  onClick={() => handleImageClick(idx)}
                />
              </li>
            ))}
          </ul>
        </div>
      </div>
    </>
  );
}
