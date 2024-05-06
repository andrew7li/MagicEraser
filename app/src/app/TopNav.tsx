import React from "react";
import styles from "./topnav.module.scss";
import { IoLogoGithub } from "react-icons/io";

export default function TopNav() {
  return (
    <div className={styles.topNavigation}>
      <div className={styles.logoContainer}>
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
