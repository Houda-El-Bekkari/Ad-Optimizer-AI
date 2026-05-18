import { Injectable } from '@angular/core';

import type { AuthUser, LoginResponse } from './auth-api';

@Injectable({ providedIn: 'root' })
export class AuthSession {
  private readonly tokenKey = 'access_token';
  private readonly expiresAtKey = 'access_token_expires_at';
  private readonly userKey = 'user';

  setSession(response: LoginResponse): void {
    const user: AuthUser = {
      user_id: response.user_id,
      username: response.username,
      email: response.email,
      role: response.role,
    };

    localStorage.setItem(this.tokenKey, response.access_token);
    localStorage.setItem(this.expiresAtKey, response.expires_at);
    localStorage.setItem(this.userKey, JSON.stringify(user));
  }

  getToken(): string {
    if (this.isSessionExpired()) {
      this.clearSession();
      return '';
    }

    return localStorage.getItem(this.tokenKey) || '';
  }

  isAuthenticated(): boolean {
    return Boolean(this.getToken());
  }

  currentUser(): AuthUser | null {
    const rawUser = localStorage.getItem(this.userKey);

    if (!rawUser || this.isSessionExpired()) {
      return null;
    }

    try {
      return JSON.parse(rawUser) as AuthUser;
    } catch {
      return null;
    }
  }

  clearSession(): void {
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem(this.expiresAtKey);
    localStorage.removeItem(this.userKey);
  }

  private isSessionExpired(): boolean {
    const expiresAt = localStorage.getItem(this.expiresAtKey);

    if (!expiresAt) {
      return false;
    }

    const expiresAtTime = new Date(expiresAt).getTime();

    return Number.isFinite(expiresAtTime) && expiresAtTime <= Date.now();
  }
}
