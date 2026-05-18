import { Component } from '@angular/core';

import { Router } from '@angular/router';

import { RouterLink } from '@angular/router';

import { FormsModule } from '@angular/forms';

import { NgIf } from '@angular/common';

import { AuthApi } from '../../services/auth-api';



@Component({

  selector: 'app-login',

  imports: [
    RouterLink,
    FormsModule,
    NgIf
  ],

  templateUrl: './login.html',

  styleUrl: './login.scss',
})

export class Login {

  email = '';

  password = '';

  emailFocused = false;

  pwFocused = false;

  showPw = false;

  loading = false;



  constructor(

    private authApi: AuthApi,

    private router: Router

  ) {}



  login() {

    const payload = {

      email: this.email,

      password: this.password
    };



    this.authApi.login(payload).subscribe({

next: (response: any) => {

  if (!response.success) {

    alert(response.message);

    return;
  }

  this.router.navigate([
    '/connect-platforms'
  ]);
},



      error: (error) => {

        console.error(error);

        alert('Login failed');
      }
    });
  }
}
