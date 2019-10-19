import React, { Component } from 'react';
import { Route, Switch } from 'react-router-dom';
import './App.css';
import Home from './pages/Home';
import List from './pages/List';
import reportes from './pages/reportes';
import Login from './pages/Login';
import Register from './pages/Register';
import Home2 from './pages/Home2';
import reportes2 from './pages/reportes2';

class App extends Component {
  render() {
    const App = () => (
      <div>
        <Switch>
          <Route exact path="/" component={Home} />
          <Route path="/list" component={List} />
          <Route exact path="/reportes" component={reportes} />
          <Route exact path="/Login" component={Login} />
          <Route exact path="/Register" component={Register} />
          <Route exact path="/Home2" component={Home2} />
          <Route exact path="/reportes2" component={reportes2} />
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
