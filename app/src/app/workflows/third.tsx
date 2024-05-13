import styles from "./second.module.scss";

const mockData = {
  objects: [
    {
      uuid: "a6c7a62c-7618-4c1d-a6b8-13d984ddc142",
      objectType: "bicycle",
      topLeft: {
        x: 1500,
        y: 681,
      },
      topRight: {
        x: 1585,
        y: 681,
      },
      bottomRight: {
        x: 1585,
        y: 681,
      },
      bottomLeft: {
        x: 1500,
        y: 681,
      },
    },
    {
      uuid: "043f9838-564e-4495-881b-d1fca9a8fefb",
      objectType: "bicycle",
      topLeft: {
        x: 1380,
        y: 681,
      },
      topRight: {
        x: 1442,
        y: 681,
      },
      bottomRight: {
        x: 1442,
        y: 681,
      },
      bottomLeft: {
        x: 1380,
        y: 681,
      },
    },
    {
      uuid: "fa55792b-eb47-44a7-a871-be27fc6d83b5",
      objectType: "potted plant",
      topLeft: {
        x: 1540,
        y: 791,
      },
      topRight: {
        x: 1597,
        y: 791,
      },
      bottomRight: {
        x: 1597,
        y: 791,
      },
      bottomLeft: {
        x: 1540,
        y: 791,
      },
    },
    {
      uuid: "720007ba-557a-474e-ab8e-44a09e2a5917",
      objectType: "bicycle",
      topLeft: {
        x: 390,
        y: 673,
      },
      topRight: {
        x: 505,
        y: 673,
      },
      bottomRight: {
        x: 505,
        y: 673,
      },
      bottomLeft: {
        x: 390,
        y: 673,
      },
    },
  ],
};

export default function Third() {
  return (
    <>
      <div className={styles.leftContainer}>Fill me in!</div>
      <div className={styles.rightContainer}>
        <div>Image Segments</div>
        {mockData.objects.map((element) => (
          <div key={element.uuid}>hello</div>
        ))}
      </div>
    </>
  );
}
