import React from "react";
import styles from "./topnav.module.scss";
import { IoLogoGithub } from "react-icons/io";
import "@uploadthing/react/styles.css";

type TopNavProps = {
  setWorkflow: (newWorkflow: number) => void;
};

export default function TopNav(props: TopNavProps) {
  const { setWorkflow } = props;

  return (
    <div className={styles.topNavigation}>
      <div className={styles.logoContainer} onClick={() => setWorkflow(0)}>
        <img src="logo-transparent.png" className={styles.logo} />
        Magic Eraser
      </div>
      <div className={styles.space} />
      <a href="https://github.com/andrew7li/MagicEraser" target="_blank">
        <IoLogoGithub size={25} />
      </a>
    </div>
  );
}
