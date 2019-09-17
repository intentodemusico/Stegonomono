import React from 'react';
import { Link } from 'react-router-dom';

const TopicList = ({ match }) => {
  console.log(match.url)
  return (
    <div>
      <h3>Indice</h3>
      <ul>
        <li>
          <Link to={`${match.url}/Topic1`}>Item 1</Link>
        </li>
        <li>
          <Link to={`${match.url}/Topic2`}>Item 2</Link>
        </li>
        <li>
          <Link to={`${match.url}/Topic3`}>Item 3</Link>
        </li>
      </ul>
    </div>
  );
};

export default TopicList;
