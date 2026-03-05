import './App.css'

/** Позже с API: url картинки уже с пометками, результат проверки */
const STATIC_IMAGE = 'https://picsum.photos/seed/defect-check/800/500'
const STATIC_RESULT: 'defect' | 'ok' = 'defect'

function App() {
  const hasDefect = STATIC_RESULT === 'defect'

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">Проверка изделия</h1>
      </header>

      <main className="app__main">
        <div className="viewer">
          <div className="viewer__frame">
            <img
              className="viewer__image"
              src={STATIC_IMAGE}
              alt="Изделие с пометками"
            />
          </div>

          <div className="result">
            <span className="result__label">Результат:</span>
            <span
              className={
                hasDefect ? 'result__value result__value--defect' : 'result__value result__value--ok'
              }
            >
              {hasDefect ? 'Брак' : 'Нет брака'}
            </span>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
