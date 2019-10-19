import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import './home.css';
import PieChart from 'react-minimal-pie-chart';

class reportes extends Component {
  render() {
    return (
      <div>
        <div className="header">
          <div className="boton">
            <form action="/Main">
              <input type="submit" value="Back to Main" />
            </form>
          </div>
          <h1>STEGONOMONO</h1>
        </div>

        <div className="texto">
          <p>Reporte de su Imagen:</p>
          <PieChart
            data={[
              { title: 'Seguro', value: 10, color: '#417abf' },
              { title: 'Riesgo', value: 15, color: '#d79050' },
            ]}
          />
          <div>
            <h1>RIESGO: ALTO</h1>
          </div>
          <div id="chartdiv" />
        </div>
      </div>
    );
  }
}
export default reportes;
