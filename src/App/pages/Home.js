import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import './home.css';


class Home extends Component {
  render() {
    return (
    	<div>
    		<div className="header">
      			<h1>STEGONOMONO</h1>
    		</div>

    		<div className="texto">
    		<p>Bienvenido a STEGONOMONO. Esta herramienta permite detectar imágenes, que hallan sido alteradas
    		usando esteganografía. Adjunte la imagen que desea escanear y presione el botón de Enviar para empezar el análisis.</p>
    		</div>

    		<div className="boton">
    			<form action="/reportes">
  					<input type="file" name="myFile"/><br/><br/>
  					<input type="submit"/>
				</form>
    		</div>

   		 </div>

    );
  }
}
export default Home;