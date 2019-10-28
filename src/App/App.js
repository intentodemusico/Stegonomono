import React, { Component } from 'react';
import { Route, Switch } from 'react-router-dom';
import './App.css';
import Home from './pages/Home';
import List from './pages/List';
import Reportes from './pages/Reportes';
import Login from './pages/Login';
import Register from './pages/Register';

class App extends Component {
  render() {
    const App = () => (
      <div>
        <Switch>
          <Route exact path="/" component={Login} />
          <Route path="/list" component={List} />
          <Route exact path="/reportes" component={Reportes} />
          <Route exact path="/Login" component={Login} />
          <Route exact path="/Register" component={Register} />
          <Route exact path="/Home" component={Home} />
        </Switch>
      </div>
    );
    return (
      <Switch>
        <App />
      </Switch>
    );
  }
}

export default App;
