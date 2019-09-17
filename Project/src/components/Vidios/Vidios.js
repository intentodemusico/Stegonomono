import React from 'react';
import { Link } from 'react-router-dom';

const Vidios = () => {
  return (
    <div>
      <img
        alt="trabajando"
        src="https://media1.tenor.com/images/8b19feb0d9eec43509283e74917a7fe9/tenor.gif?itemid=10405354"
      />
      <iframe
        title="videocolas"
        width="560"
        height="315"
        src="https://www.youtube.com/embed/p5gokwGSJ-0?controls=0"
        frameborder="0"
        allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen
      />
    </div>
  );
};

export default Vidios;
