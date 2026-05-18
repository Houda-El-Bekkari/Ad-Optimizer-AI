import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Router } from '@angular/router';
import { Observable, tap } from 'rxjs';

import { environment } from '../../environments/environment';
import { AuthSession } from './auth-session';

export interface LoginPayload {
  email: string;
  password: string;
}

export interface SignupPayload {
  username: string;
  email: string;
  password: string;
  role?: string;
}

export interface AuthUser {
  user_id: number;
  username: string;
  email: string;
  role: string;
}

export interface LoginResponse extends AuthUser {
  success: boolean;
  message: string;
  access_token: string;
  token_type: string;
  expires_at: string;
}

@Injectable({ providedIn: 'root' })
export class AuthApi {
  private readonly apiUrl = environment.apiUrl;

  constructor(
    private readonly http: HttpClient,
    private readonly router: Router,
    private readonly session: AuthSession,
  ) {}

  login(payload: LoginPayload): Observable<LoginResponse> {
    return this.http.post<LoginResponse>(`${this.apiUrl}/login`, payload).pipe(
      tap((response) => {
        if (response.success && response.access_token) {
          this.session.setSession(response);
        }
      }),
    );
  }

  signup(payload: SignupPayload): Observable<unknown> {
    return this.http.post(`${this.apiUrl}/signup`, payload);
  }

  getToken(): string {
    return this.session.getToken();
  }

  isAuthenticated(): boolean {
    return this.session.isAuthenticated();
  }

  logout(): void {
    this.session.clearSession();
    void this.router.navigate(['/login']);
  }

  currentUser(): AuthUser | null {
    return this.session.currentUser();
  }
}
