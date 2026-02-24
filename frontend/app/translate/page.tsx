export default function TranslateLegacyPage() {
  return (
    <main className="mx-auto max-w-3xl p-8">
      <h1 className="text-2xl font-bold">Legacy Translate Flow</h1>
      <p className="mt-3 text-sm text-slate-600">
        The primary UX has moved to term-level annotation on the home page.
      </p>
      <p className="mt-2 text-sm text-slate-600">
        You can still call the backend translate API directly at /translate.
      </p>
      <a className="mt-4 inline-block text-blue-600 underline" href="/">
        Go to Annotate
      </a>
    </main>
  );
}
