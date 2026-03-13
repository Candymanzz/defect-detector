/**
 * Конфигурация фронтенда. API бэкенда задаётся через переменную окружения при сборке.
 */
const API_BASE_URL =
  typeof import.meta.env?.VITE_API_URL === 'string' && import.meta.env.VITE_API_URL !== ''
    ? import.meta.env.VITE_API_URL.replace(/\/$/, '')
    : 'http://localhost:8000'

export const config = {
  /** Базовый URL API (Patchcore analyze, health и т.д.) */
  apiBaseUrl: API_BASE_URL,
} as const
