import '@fortawesome/fontawesome-free/css/all.min.css';
import 'bootstrap-css-only/css/bootstrap.min.css';
import 'mdbreact/dist/css/mdb.css';

const express = require('express');
const path = require('path');

const app = express();

// Sevri archivos estáticos desde la app con React
app.use(express.static(path.join(__dirname, 'client/build')));

// Un endpoint del API que devuelve una lista
app.get('/api/getList', (req,res) => {
	let list = findAll()
	res.json(list);
	console.log('Enviada lista con datos');
});

// Handler universal
app.get('*', (req,res) =>{
	res.sendFile(path.join(__dirname+'/client/build/index.html'));
});

const port = process.env.PORT || 5000;
app.listen(port);

console.log('Express escuchando en el puerto ' + port);
