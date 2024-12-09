import unittest
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_all_tests():
    """Esegue tutti i test e genera un report"""
    # Trova tutti i test
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(str(Path(__file__).parent), pattern="test_*.py")
    
    # Prepara il report
    results = {
        'timestamp': datetime.now().isoformat(),
        'results': {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0
        },
        'failures': [],
        'errors': []
    }
    
    # Esegui i test
    print("\n=== ESECUZIONE TEST COMPLETI ===\n")
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)
    
    # Aggiorna risultati
    results['results']['total'] = test_result.testsRun
    results['results']['passed'] = test_result.testsRun - len(test_result.failures) - len(test_result.errors)
    results['results']['failed'] = len(test_result.failures)
    results['results']['errors'] = len(test_result.errors)
    results['results']['skipped'] = len(test_result.skipped)
    
    # Registra fallimenti
    for failure in test_result.failures:
        results['failures'].append({
            'test': str(failure[0]),
            'message': str(failure[1])
        })
        
    # Registra errori
    for error in test_result.errors:
        results['errors'].append({
            'test': str(error[0]),
            'message': str(error[1])
        })
    
    # Salva report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'test_report_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Stampa sommario
    print("\n=== SOMMARIO TEST ===")
    print(f"Test totali: {results['results']['total']}")
    print(f"Test passati: {results['results']['passed']}")
    print(f"Test falliti: {results['results']['failed']}")
    print(f"Errori: {results['results']['errors']}")
    print(f"Test saltati: {results['results']['skipped']}")
    print(f"\nReport salvato in: {report_file}")
    
    # Restituisci successo/fallimento
    return len(test_result.failures) + len(test_result.errors) == 0

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
