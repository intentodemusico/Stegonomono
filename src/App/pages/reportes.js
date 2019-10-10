import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import './home.css';

class reportes extends Component {

  render() {
    return (
    	<div>
    		<div className="header">
      			<h1>STEGONOMONO</h1>
    		</div>

    		<div className="texto">
    		<p>Reporte de su Imagen:</p>
        <div id="chartdiv"></div>
    		</div>
        
   		 </div>

    );
  }
}
export default reportes;