import styles from "./second.module.scss";

type SecondProps = {
  setWorkflow: (newWorkflow: number) => void;
  file: File;
};

export default function Second(props: SecondProps) {
  const { setWorkflow, file } = props;
  return (
    <>
      <div className={styles.leftContainer}>Fill me in!</div>
      <div className={styles.rightContainer}>Fill me!</div>
    </>
  );
}
