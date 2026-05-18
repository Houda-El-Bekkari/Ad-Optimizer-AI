import { Component } from '@angular/core';

import { Router } from '@angular/router';

import { RouterLink } from '@angular/router';

import { FormsModule } from '@angular/forms';

import { NgIf } from '@angular/common';

import { AuthApi } from '../../services/auth-api';



@Component({

  selector: 'app-signup',

  imports: [
    RouterLink,
    FormsModule,
    NgIf
  ],

  templateUrl: './signup.html',

  styleUrl: './signup.scss',
})

export class Signup {

  username = '';

  email = '';

  password = '';

  confirmPassword = '';

  showPw = false;

  loading = false;



  constructor(

    private authApi: AuthApi,

    private router: Router

  ) {}



  signup() {

    const payload = {

      username: this.username,

      email: this.email,

      password: this.password,

      role: 'marketing_user'
    };



    this.authApi.signup(payload).subscribe({

      next: (response: any) => {

        console.log(response);



        alert('Account created successfully');



        this.router.navigate([
          '/login'
        ]);
      },



      error: (error) => {

        console.error(error);

        alert('Signup failed');
      }
    });
  }
}
