import styles from "./test.module.scss";

import { MdOutlineFileUpload } from "react-icons/md";
import TopNav from "./TopNav";

export default function Home() {
  return (
    <div className={styles.body}>
      <TopNav />
      <div className={styles.headerContainer}>
        <p className={styles.headerText}>
          <span>Remove Objects with</span> AI Magic Eraser
        </p>
      </div>
      <div className={styles.imageContainer}>
        <div className={styles.imageDemoContainer}>
          <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />
        </div>
        <div className={styles.imageUploadContainer}>
          <input
            type="file"
            id="inputFile"
            style={{ display: "none" }} // Hide the input but keep it functional
            accept="image/jpeg, image/png, image/webp, image/bmp" // Specify accepted file types
            multiple // Allow multiple file uploads if necessary
            // onChange={handleFileUpload} // Implement a function to handle file uploads
          />
          <div className={styles.uploadImageDragAndDropContainer}>
            <label htmlFor="inputFile" className={styles.uploadImageTextLabel}>
              <span>
                <MdOutlineFileUpload size={20} />
              </span>
              <ul>
                <li>Drag or drop images here</li>
                <li>JPG, PNG, JPEG, WEBP, BMP· Max size 5Mb</li>
              </ul>
            </label>

            <p>
              *All files are stored privately & encrypted. Only you will see
              them.
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
      <div className={styles.descriptionContainer}>
        <strong>Instantly modify your images to create a visual impact</strong>
        <br />
        <p>
          Experience the magic of removing objects beyond what is traditionally
          capable
        </p>
      </div>
    </div>
  );
}
