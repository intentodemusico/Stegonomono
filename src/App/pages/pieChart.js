import React from 'react';
import { Pie } from 'react-chartjs-2';
import { MDBContainer } from 'mdbreact';

class PieChart extends React.Component {
  state = {
    dataPie: {
      labels: ['Riesgo', 'Seguro'],
      datasets: [
        {
          data: [300, 50],
          backgroundColor: ['#F7464A', '#46BFBD'],
          hoverBackgroundColor: ['#FF5A5E', '#5AD3D1'],
        },
      ],
    },
  };

  render() {
    return (
      <MDBContainer>
        <Pie data={this.state.dataPie} options={{ responsive: true }} />
      </MDBContainer>
    );
  }
}

export default PieChart;
