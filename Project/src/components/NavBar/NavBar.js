import React from 'react';
import { Link } from 'react-router-dom';

import './NavBar.css';

const NavBar = () => {
  return (
    <div>
      <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
        <Link className="navbar-brand" to="/Home">
          SITIO PARCIAL
        </Link>
        <ul className="nav">
          <li className="nav-item active">
            <Link className="nav-link" to="/Home">
              Home
            </Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/About">
              Acerca de
            </Link>
            ={' '}
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/Topics">
              Indice
            </Link>
            =
          </li>
        </ul>
        <hr />
      </nav>
    </div>
  );
};

export default NavBar;
