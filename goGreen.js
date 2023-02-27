const jsonfile = require('jsonfile');
const moment = require('moment');
const simpleGit = require('simple-git');

const FILE_PATH = './data.json';
const git = simpleGit();
const TOTAL_COMMITS = 40;

const buildCommitDates = () => {
  const start = moment('2023-01-03T12:00:00');
  const dates = [];

  for (let i = 0; i < TOTAL_COMMITS; i += 1) {
    let date = start.clone().add(i * 8, 'days');

    if (date.isoWeekday() > 5) {
      date = date.add(2, 'days');
    }

    dates.push(date.format());
  }

  return dates;
};

const commitDates = buildCommitDates();
let index = 0;

const makeCommit = async () => {
  if (index >= commitDates.length) {
    console.log('Tüm commitler tamamlandı.');
    return;
  }

  const DATE = commitDates[index];
  index += 1;

  const data = {
    date: DATE,
    commitIndex: index
  };

  console.log(`Commit yapılıyor [${index}/${TOTAL_COMMITS}]: ${DATE}`);

  await git.addConfig('user.name', 'GitHub Copilot');
  await git.addConfig('user.email', 'copilot@example.com');

  jsonfile.writeFile(FILE_PATH, data, async () => {
    try {
      await git.add([FILE_PATH, 'goGreen.js']);
      await git.commit(`commit-${index} (${DATE})`, { '--date': DATE });
      await makeCommit();
    } catch (error) {
      console.error('Commit sırasında hata:', error);
    }
  });
};

makeCommit();
