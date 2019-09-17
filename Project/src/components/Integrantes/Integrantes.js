import React from 'react';
import { Link } from 'react-router-dom';

const Integrantes = () => {
  return (
    <div>
      <h3>Integrantes</h3>
      <ul>
        <li>
          Andrea Claro :)
          <img
            alt="cat"
            src="https://66.media.tumblr.com/47776a0dcac8e5546884bbb3917faebe/tumblr_pseva5AQTP1vk0a3q_540.jpg"
            width="8%"
          />
        </li>
        <li>
          Alberto García :D
          <img
            alt="doge"
            src="http://www.rw-designer.com/icon-image/15661-256x256x32.png"
            width="10%"
          />
        </li>
      </ul>
    </div>
  );
};

export default Integrantes;
